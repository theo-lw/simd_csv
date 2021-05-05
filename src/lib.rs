/* Steps:
 *  1. Identify the quoted regions (done using the poly mult trick)
 *  2. Identify the delimiters and the CRLF's
 *  3. Check that fields that contain quotes are contained within quotes (can be done through some
 *     bit magic)
 *  4. Check that each line has the same # of fields
 *
 * Design:
 *  1. The CsvReader should read pages from a reader & decode each page that was read.
 *  2. The CsvReader should be convertible into an iterator over each line of the file.
 *  3. We should be able to iterate over each field in the line.
 *  4. Lazily decode each field into a string slice.
 *  5. This means lazily check that the field is UTF-8?
 *
 * Considerations:
 *  1. Paging - processing pages lends itself to unpredictable performance spikes.
 *     Fetching a new page poses some issues. Namely, what if you have a half-parsed line?
 *     This line has to be moved to the front of the internal buffer.
 *
 * Things to add:
 *  1. Deserializing each line
 *  2. Giving users the option to skip the header?
 */

pub mod simd_csv {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline(always)]
    fn get_char_mask(csv_slice: __m256i, ch: u8) -> i32 {
        unsafe {
            let mask = _mm256_set1_epi8(ch as i8);
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(mask, csv_slice))
        }
    }

    #[inline(always)]
    fn find_quoted_regions(csv_slice: __m256i, in_quoted_region: i32) -> (i32, i32, i32) {
        let csv_quotes = get_char_mask(csv_slice, QUOTE);
        unsafe {
            let all_ones = _mm_set1_epi8(-1);
            let quoted_region = _mm_cvtsi128_si32(_mm_clmulepi64_si128(
                _mm_set_epi32(0, 0, 0, csv_quotes ^ in_quoted_region),
                all_ones,
                0,
            ));
            let quoted_region_with_quotes = quoted_region | csv_quotes;
            let end_in_quoted_region = get_last_bit(quoted_region);
            (quoted_region_with_quotes, csv_quotes, end_in_quoted_region)
        }
    }

    #[inline(always)]
    fn get_last_bit(num: i32) -> i32 {
        ((num as u32) >> 31) as i32
    }

    #[inline(always)]
    fn is_bitset_a_subset(subset: u32, superset: u32) -> bool {
        (subset & superset) == subset
    }

    #[inline(always)] // check if there are any fields containing quote/CR/LFs that are not surrounded by quotes.
    fn exists_missing_quotes(
        quoted_regions: i32,
        delims_and_crlfs: i32,
        crs: i32, // carriage returns
        lfs: i32, // line feeds
        in_crlf: i32,
        in_quoted_region: i32,
        ended_delim: bool,
        ended_quoted_region: bool,
    ) -> bool {
        let quoted_regions = quoted_regions as u32;
        let delims_and_crlfs = delims_and_crlfs as u32;
        let crs = crs as u32;
        let lfs = lfs as u32;
        let in_crlf = in_crlf as u32;
        let prev_quoted_region_ended = !ended_quoted_region || (delims_and_crlfs & 1 == 1);
        let starting_quoted_region_preceeded_by_delim =
            ended_delim || in_quoted_region == 1 || quoted_regions & 1 != 1;
        let left_right_of_quoted_regions =
            ((quoted_regions << 1) | (quoted_regions >> 1)) & !quoted_regions;
        let all_crs_followed_by_lfs = (in_crlf == lfs & 1) && is_bitset_a_subset(crs << 1, lfs);
        let all_lfs_preceeded_by_crs = is_bitset_a_subset(lfs >> 1, crs);
        !(prev_quoted_region_ended
            && starting_quoted_region_preceeded_by_delim
            // check that spots left & right of each quoted region are occupied by delims / crlfs
            && (left_right_of_quoted_regions & delims_and_crlfs == left_right_of_quoted_regions)
            && all_lfs_preceeded_by_crs
            && all_crs_followed_by_lfs)
    }

    #[derive(Debug)]
    pub enum ParseError {
        InconsistentColumnCount,
        MissingQuotes,
    }

    #[derive(Debug, PartialEq, Clone, Copy)]
    enum ParseState {
        Start,
        Delim,
        NonQuotedField,
        QuotedField,
        EndQuote,
        CarriageReturn,
        LineFeed,
    }

    const QUOTE: u8 = b'"';
    const CARRIAGE_RETURN: u8 = b'\r';
    const LINE_FEED: u8 = b'\n';

    #[derive(Debug, PartialEq)]
    pub struct Token {
        pub pos: usize,
        pub quotes_before: usize,
        pub token_type: TokenType,
    }

    #[derive(Debug, PartialEq)]
    pub enum TokenType {
        Delim,
        CarriageReturn,
        LineFeed,
        End,
    }

    pub fn decode_csv_structure(csv: &[u8], delim: u8) -> Result<Vec<Token>, ParseError> {
        const STEP: usize = 256 / 8;
        let mut result = Vec::new();
        let mut csv_idx = 0;
        let mut in_quoted_region: i32 = 0; // 0 when not in quoted region, 1 otherwise. Used to "carry" quoted regions from one SIMD vector to the next SIMD vector
        let mut in_crlf: i32 = 0; // 1 when the previous SIMD vector ended with a CR, 0 otherwise
        let mut ended_delim: bool = true; // if the last SIMD vector ended with BOF/newline/comma
        let mut ended_quoted_region: bool = false; // if the last SIMD vector ended with the ending of a quoted region
        let mut parse_error: bool = false;
        let mut num_carried_quotes = 0; // quotes at the end of the previous SIMD vector
        unsafe {
            while csv_idx + STEP <= csv.len() {
                #[allow(clippy::cast_ptr_alignment)]
                // Intel's documentation says that unaligned pointers are fine here
                let csv_slice = _mm256_loadu_si256(csv[csv_idx..].as_ptr() as *const __m256i);
                let (quoted_region, mut quotes, end_in_quoted_region) =
                    find_quoted_regions(csv_slice, in_quoted_region);
                let mut crs = get_char_mask(csv_slice, CARRIAGE_RETURN) & !quoted_region;
                let mut lfs = get_char_mask(csv_slice, LINE_FEED) & !quoted_region;
                let delims = get_char_mask(csv_slice, delim) & !quoted_region;
                let mut all_structural_chars = delims | crs | lfs;

                parse_error |= exists_missing_quotes(
                    quoted_region,
                    all_structural_chars,
                    crs,
                    lfs,
                    in_crlf,
                    in_quoted_region,
                    ended_delim,
                    ended_quoted_region,
                );

                ended_delim = get_last_bit(all_structural_chars) == 1;

                // retrieve the indices of the delimiters
                let mut shift_cnt = 0;
                for _ in 0.._popcnt32(all_structural_chars) {
                    let next_idx = _tzcnt_u32(all_structural_chars as u32) as usize;
                    crs >>= next_idx;
                    lfs >>= next_idx;
                    let quote_mask: i32 = if next_idx == 0 {
                        0
                    } else {
                        ((1 << next_idx) as i32).wrapping_sub(1)
                    };
                    let quotes_before =
                        (_popcnt32(quotes & quote_mask) + num_carried_quotes) as usize;
                    let pos = csv_idx + next_idx + shift_cnt;
                    let token_type = if crs & 1 == 1 {
                        TokenType::CarriageReturn
                    } else if lfs & 1 == 1 {
                        TokenType::LineFeed
                    } else {
                        TokenType::Delim
                    };
                    result.push(Token {
                        pos,
                        quotes_before,
                        token_type,
                    });
                    all_structural_chars >>= next_idx;
                    all_structural_chars >>= 1;
                    quotes >>= next_idx;
                    quotes >>= 1;
                    crs >>= 1;
                    lfs >>= 1;
                    shift_cnt += next_idx + 1;
                    num_carried_quotes = 0;
                }

                num_carried_quotes = if shift_cnt == 0 {
                    _popcnt32(quotes)
                } else {
                    _popcnt32(quotes & ((1i32 << (32 - shift_cnt)).wrapping_sub(1)))
                };
                csv_idx += STEP;
                in_quoted_region = end_in_quoted_region;
                ended_quoted_region = get_last_bit(quoted_region) == 1 && end_in_quoted_region == 0;
                in_crlf = get_last_bit(crs);
            }
        }

        // Parse the remaining characters using a DFA (this is at most 31 characters, so the
        // branches shouldn't have too big of an impact)
        let mut parse_state = if csv_idx == 0 {
            ParseState::Start
        } else if in_quoted_region == 1 {
            ParseState::QuotedField
        } else if csv[csv_idx - 1] == QUOTE {
            ParseState::EndQuote
        } else if csv[csv_idx - 1] == delim {
            ParseState::Delim
        } else if csv[csv_idx - 1] == CARRIAGE_RETURN {
            ParseState::CarriageReturn
        } else if csv[csv_idx - 1] == LINE_FEED {
            ParseState::LineFeed
        } else {
            ParseState::NonQuotedField
        };

        for (idx, elem) in csv[csv_idx..].iter().enumerate() {
            match (parse_state, *elem) {
                (ParseState::Start, QUOTE) => parse_state = ParseState::QuotedField,
                (ParseState::Start, CARRIAGE_RETURN) => parse_state = ParseState::CarriageReturn,
                (ParseState::Start, ch) if ch == delim => parse_state = ParseState::Delim,
                (ParseState::Start, ch) if ch != delim && ch != LINE_FEED => {
                    parse_state = ParseState::NonQuotedField
                }
                (ParseState::QuotedField, QUOTE) => parse_state = ParseState::EndQuote,
                (ParseState::QuotedField, _) => {}
                (ParseState::EndQuote, QUOTE) => parse_state = ParseState::QuotedField,
                (ParseState::EndQuote, CARRIAGE_RETURN) => parse_state = ParseState::CarriageReturn,
                (ParseState::EndQuote, ch) if ch == delim => parse_state = ParseState::Delim,
                (ParseState::Delim, QUOTE) => parse_state = ParseState::QuotedField,
                (ParseState::Delim, CARRIAGE_RETURN) => parse_state = ParseState::CarriageReturn,
                (ParseState::Delim, ch) if ch != delim && ch != LINE_FEED => {
                    parse_state = ParseState::NonQuotedField
                }
                (ParseState::CarriageReturn, LINE_FEED) => parse_state = ParseState::LineFeed,
                (ParseState::LineFeed, QUOTE) => parse_state = ParseState::QuotedField,
                (ParseState::LineFeed, CARRIAGE_RETURN) => parse_state = ParseState::CarriageReturn,
                (ParseState::LineFeed, ch) if ch == delim => parse_state = ParseState::Delim,
                (ParseState::LineFeed, ch) if ch != delim && ch != LINE_FEED => {
                    parse_state = ParseState::NonQuotedField
                }
                (ParseState::NonQuotedField, CARRIAGE_RETURN) => {
                    parse_state = ParseState::CarriageReturn
                }
                (ParseState::NonQuotedField, ch) if ch == delim => parse_state = ParseState::Delim,
                (ParseState::NonQuotedField, ch)
                    if ch != delim && ch != LINE_FEED && ch != QUOTE => {}
                (_, _) => return Err(ParseError::MissingQuotes),
            };

            if *elem == QUOTE {
                num_carried_quotes += 1;
            }

            if parse_state != ParseState::Delim
                && parse_state != ParseState::CarriageReturn
                && parse_state != ParseState::LineFeed
            {
                continue;
            }

            let token_type = if parse_state == ParseState::Delim {
                TokenType::Delim
            } else if parse_state == ParseState::CarriageReturn {
                TokenType::CarriageReturn
            } else {
                TokenType::LineFeed
            };

            result.push(Token {
                pos: csv_idx + idx,
                quotes_before: num_carried_quotes as usize,
                token_type,
            });

            num_carried_quotes = 0;
        }

        if parse_error
            || parse_state == ParseState::QuotedField
            || parse_state == ParseState::CarriageReturn
        {
            return Err(ParseError::MissingQuotes);
        }

        result.push(Token {
            pos: csv.len(),
            quotes_before: num_carried_quotes as usize,
            token_type: TokenType::End,
        });

        Ok(result)
    }

    #[derive(Clone, Copy, Debug)]
    pub enum CsvItemType {
        Field,
        CRLF,
    }

    pub struct CsvItem<'a> {
        span: &'a mut [u8],
        length: usize,
        decoded: std::cell::Cell<bool>,
        pub item_type: CsvItemType,
    }

    impl<'a> CsvItem<'a> {
        fn unquote(&mut self) {
            if self.decoded.get() {
                return;
            }
            let mut write_idx = 0;
            let mut num_quotes = 0;
            for idx in 0..self.span.len() {
                if self.span[idx] == QUOTE {
                    num_quotes += 1;
                }
                if num_quotes == 2 {
                    num_quotes = 0;
                    continue;
                }
                self.span[write_idx] = self.span[idx];
                write_idx += 1;
            }
            self.length = write_idx;
            self.decoded.set(true);
        }

        pub fn decode(&'a mut self) -> Option<&'a [u8]> {
            self.unquote();
            match self.item_type {
                CsvItemType::Field => Some(&self.span[0..self.length]),
                CsvItemType::CRLF => None,
            }
        }

        pub fn decode_mut(&'a mut self) -> Option<&'a mut [u8]> {
            self.unquote();
            match self.item_type {
                CsvItemType::Field => Some(&mut self.span[0..self.length]),
                CsvItemType::CRLF => None,
            }
        }
    }

    pub fn extract_csv<'a>(csv: &'a mut [u8], delim: u8) -> Result<Vec<CsvItem<'a>>, ParseError> {
        let delim_indices = decode_csv_structure(csv, delim);
        let mut starting_idx = usize::MAX;
        let mut result = Vec::new();
        let mut csv = csv;
        let mut consumed = 0;
        for Token {
            pos,
            quotes_before,
            token_type,
        } in delim_indices?
        {
            starting_idx = starting_idx.wrapping_add(1);

            // judging by the godbolt output, this should not compile down to a branch
            starting_idx += if quotes_before > 0 { 1 } else { 0 };
            let ending_idx = pos.saturating_sub(if quotes_before > 0 { 1 } else { 0 });

            let (_, span) = csv.split_at_mut(starting_idx - consumed);
            let (span, end) = span.split_at_mut(ending_idx - starting_idx);

            let item_type = match token_type {
                TokenType::LineFeed => CsvItemType::CRLF,
                _ => CsvItemType::Field,
            };

            let length = span.len();

            result.push(CsvItem {
                span,
                length,
                decoded: std::cell::Cell::new(quotes_before <= 2),
                item_type,
            });

            consumed = ending_idx;
            starting_idx = pos;
            csv = end;
        }
        Ok(result)
    }

    #[cfg(test)]
    mod tests {
        use super::{decode_csv_structure, extract_csv, Token, TokenType};

        #[test]
        fn find_extract_csv_works() {
            let mut csv = *b"a,\"b\"\",c\", d ,e,";
            let mut extracted = extract_csv(&mut csv, b',').unwrap();
            match extracted.as_mut_slice() {
                [ref mut first, ref mut second, ref mut third, ref mut fourth, ref mut fifth] => {
                    assert_eq!(first.decode().unwrap(), b"a");
                    assert_eq!(second.decode().unwrap(), b"b\",c");
                    assert_eq!(third.decode().unwrap(), b" d ");
                    assert_eq!(fourth.decode().unwrap(), b"e");
                    assert_eq!(fifth.decode().unwrap(), b"");
                }
                _ => panic!("Slice doesn't match"),
            }
        }

        #[test]
        fn decode_csv_structure_works() {
            assert_eq!(
                decode_csv_structure(b"asd,s,\"s\"\",\r\ns\",b,a,\r\naaaaaaaaaaaaaaa,", b',')
                    .unwrap(),
                vec![
                    Token {
                        pos: 3,
                        quotes_before: 0,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 5,
                        quotes_before: 0,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 15,
                        quotes_before: 4,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 17,
                        quotes_before: 0,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 19,
                        quotes_before: 0,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 20,
                        quotes_before: 0,
                        token_type: TokenType::CarriageReturn
                    },
                    Token {
                        pos: 21,
                        quotes_before: 0,
                        token_type: TokenType::LineFeed
                    },
                    Token {
                        pos: 37,
                        quotes_before: 0,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 38,
                        quotes_before: 0,
                        token_type: TokenType::End
                    }
                ]
            );
            assert_eq!(
                decode_csv_structure(
                    b"0123456789ABCDEF0123456789ABCDE\r\n\
                  0123456789ABCDEF0123456789ABCDEF",
                    b',',
                )
                .unwrap(),
                vec![
                    Token {
                        pos: 31,
                        quotes_before: 0,
                        token_type: TokenType::CarriageReturn
                    },
                    Token {
                        pos: 32,
                        quotes_before: 0,
                        token_type: TokenType::LineFeed
                    },
                    Token {
                        pos: 65,
                        quotes_before: 0,
                        token_type: TokenType::End
                    }
                ]
            );
            assert_eq!(
                decode_csv_structure(
                    b"0123456789ABCDEF0123456789A,\"DE\r\n\
                  012\",56789ABCDEF0123456789ABCDEF",
                    b',',
                )
                .unwrap(),
                vec![
                    Token {
                        pos: 27,
                        quotes_before: 0,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 37,
                        quotes_before: 2,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 65,
                        quotes_before: 0,
                        token_type: TokenType::End
                    }
                ]
            );
            assert_eq!(
                decode_csv_structure(
                    b"0123456789ABCDEF0123456789ABCD,\"\"\
                  ,123456789ABCDEF0123456789ABCDEF",
                    b',',
                )
                .unwrap(),
                vec![
                    Token {
                        pos: 30,
                        quotes_before: 0,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 33,
                        quotes_before: 2,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 65,
                        quotes_before: 0,
                        token_type: TokenType::End
                    }
                ]
            );
            assert_eq!(
                decode_csv_structure(b"a,b,\"c,\",\r\n hi,", b',').unwrap(),
                vec![
                    Token {
                        pos: 1,
                        quotes_before: 0,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 3,
                        quotes_before: 0,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 8,
                        quotes_before: 2,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 9,
                        quotes_before: 0,
                        token_type: TokenType::CarriageReturn
                    },
                    Token {
                        pos: 10,
                        quotes_before: 0,
                        token_type: TokenType::LineFeed
                    },
                    Token {
                        pos: 14,
                        quotes_before: 0,
                        token_type: TokenType::Delim
                    },
                    Token {
                        pos: 15,
                        quotes_before: 0,
                        token_type: TokenType::End
                    }
                ]
            );
        }

        #[test]
        fn err_on_missing_quotes() {
            assert!(decode_csv_structure(b" \"missing\", a", b',').is_err());
            assert!(decode_csv_structure(b"helllo,\"mis", b',').is_err());
            assert!(decode_csv_structure(b"he\nlllo,\"mis\"", b',').is_err());
            assert!(
                decode_csv_structure(b"hello,world, \"missing\", aaaaaaaaaaaaaaaaa", b',').is_err()
            );
            assert!(
                decode_csv_structure(b"hello,world,\"missing\" , aaaaaaaaaaaaaaaaa", b',').is_err()
            );
            assert!(
                decode_csv_structure(b"hello,world,\"missing\",\"aaaaaaaaaaaaaaaaa", b',').is_err()
            );
            assert!(
                decode_csv_structure(b"hello,wor\rld,\"missing\",aaaaaaaaaaaaaaaaa", b',').is_err()
            );
            assert!(
                decode_csv_structure(b"hello,wor\nld,\"missing\",aaaaaaaaaaaaaaaaa", b',').is_err()
            );
            assert!(
                decode_csv_structure(b"hello,wor\n\rld,\"missing\",aaaaaaaaaaaaaaaaa", b',')
                    .is_err()
            );
            assert!(decode_csv_structure(
                b"0123456789ABCDEF0123456789ABCD,\"\" ,123456789ABCDEF0123456789ABCDEF",
                b',',
            )
            .is_err());
            assert!(decode_csv_structure(
                b"0123456789ABCDEF0123456789ABCD,\" \" ,123456789ABCDEF0123456789ABCDEF",
                b',',
            )
            .is_err());
            assert!(decode_csv_structure(
                b"0123456789ABCDEF0123456789ABCD,\"\"\"\" ,123456789ABCDEF0123456789ABCDEF",
                b',',
            )
            .is_err());
        }
    }
}
