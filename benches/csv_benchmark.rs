use criterion::{criterion_group, criterion_main, Criterion};
use csv::Reader;
use simd_csv::{extract_csv, CsvItem, CsvItemType};

const LINE: &[u8] = b"hello, world ,\"404, asdflkajsdflaksjdfasdlfkjasdasdf 303\"\"\r\n\", ,\r\n";
// const NUM_FIELDS_PER_LINE: usize = 5;
const NUM_LINES: usize = 1000;

fn bench_simd_csv(csv: &[u8]) {
    let mut csv = csv.to_owned();
    let mut result = extract_csv(&mut csv, b',').unwrap();
    let mut cnt = 0;
    for item in result.iter_mut() {
        match item.item_type {
            CsvItemType::Field => cnt += 1,
            _ => {}
        }
    }
    assert_eq!(cnt, 5001);
}

fn bench_csv(csv: &[u8]) {
    let mut rdr = csv::Reader::from_reader(csv);
    let mut cnt = 0;
    for record in rdr.records() {
        for _ in record.unwrap().iter() {
            cnt += 1;
        }
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut csv = Vec::new();
    for _ in 0..NUM_LINES {
        csv.extend(LINE);
    }
    let mut group = c.benchmark_group("csv parsing");
    group.bench_function("simd csv", |b| b.iter(|| bench_simd_csv(&csv)));
    group.bench_function("csv", |b| b.iter(|| bench_csv(&csv)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
