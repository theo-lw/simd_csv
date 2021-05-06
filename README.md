# simd\_csv

An attempt at creating a very fast CSV ([RFC 4180](https://tools.ietf.org/html/rfc4180)) parser using SIMD instructions. It was inspired by Langdale and Lemire's `simdjson` library [1]. Like `simdjson`, we divide CSV parsing into two different stages:

1. Using SIMD instructions to identify the delimiters and CRLFs that compose the CSV. 
2. Iterating through the identified delimmiters and CRLFs to extract the fields.

This library has yet to go through real-world benchmarks & testing. However, microbenchmarks show some promise:

csv parsing/simd csv    time:   [331.27 us 332.94 us 335.21 us]

csv parsing/csv         time:   [715.39 us 718.07 us 721.26 us]

## TODO

- Optional ASCII/UTF-8 validation
- Validating that the # columns is the same in each row
- Offering a paging parser for very large files
- Serde compatibility
- Adding more tests, benchmarks

## References

1. Langdale, G., & Lemire, D. (2019). Parsing gigabytes of JSON per second. The VLDB Journal, 28(6), 941-960.
