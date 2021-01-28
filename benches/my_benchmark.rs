use std::{fs::File, io::BufReader};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mysearch::index::Index;
use serde_json::Value;

fn librarian(c: &mut Criterion) {
    let mut reader =
        BufReader::new(File::open("dataset/movies.json").expect("Error opening dataset file"));
    let v: Vec<Value> = serde_json::from_reader(&mut reader).expect("Error reading json dataset");
    Index::create_index(v);
    let mut librarian = Index::open().unwrap();

    c.bench_function("search 'the'", |b| {
        b.iter(|| librarian.search(&black_box("the".to_string()), 10))
    });

    c.bench_function("search 'th'", |b| {
        b.iter(|| librarian.search(&black_box("th".to_string()), 10))
    });

    c.bench_function("search 'star'", |b| {
        b.iter(|| librarian.search(&black_box("star".to_string()), 10))
    });

    c.bench_function("search 'avengel'", |b| {
        b.iter(|| librarian.search(&black_box("avengel".to_string()), 10))
    });

    c.bench_function("search 'Star wars'", |b| {
        b.iter(|| librarian.search(&black_box("Star wars".to_string()), 10))
    });

    c.bench_function("search 'The returm of the jedi'", |b| {
        b.iter(|| librarian.search(&black_box("The returm of the jedi".to_string()), 10))
    });

    c.bench_function("search 'Imterstelkar'", |b| {
        b.iter(|| librarian.search(&black_box("Imterstelkar".to_string()), 10))
    });

    c.bench_function("search 'inglorius bastards'", |b| {
        b.iter(|| librarian.search(&black_box("inglorius bastards".to_string()), 10))
    });
}

criterion_group!(benches, librarian);
criterion_main!(benches);
