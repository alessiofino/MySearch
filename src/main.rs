use std::{char, fs::File, io::BufReader, time::Instant};

use clap::{App, Arg};
use log::error;
use mysearch::index::Index;
use serde_json::Value;

use ncurses::*;

extern crate jemallocator;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

struct CursedPrinter {
    attributes: Vec<String>,
}

impl CursedPrinter {
    fn new(attr: Option<&str>) -> Self {
        let attributes = if let Some(attr) = attr {
            attr.split(",").map(|s| s.to_owned()).collect()
        } else {
            vec![]
        };
        Self { attributes }
    }

    fn print(&self, value: Value) {
        if self.attributes.is_empty() || !value.is_object() {
            addstr(&format!("{:#}", value));
        } else {
            let obj = value.as_object().unwrap();
            for attr in &self.attributes {
                if let Some(value) = obj.get(attr) {
                    attron(A_BOLD());
                    addstr(&attr);
                    attroff(A_BOLD());
                    addstr(" : ");
                    addstr(&value.to_string());
                    addstr("\n");
                }
            }
        }
    }
}
fn main() {
    let matches = App::new("MySearch")
        .version("1.0")
        .author("Alessio Fino. <alessio.fino8421@gmail.com>")
        .about("Command line utility to perform text search in JSON documents")
        .arg(
            Arg::with_name("index")
                .long("index")
                .value_name("JSON_FILE")
                .help("Index a JSON document")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("no-curses")
                .long("no-curses")
                .value_name("SEARCH_QUERY")
                .help("Do a single search without ncurses TUI")
                .takes_value(true),
        )
        .arg(
            Arg::with_name("attributes")
                .long("attributes")
                .short("a")
                .value_name("ATTRIBUTE_LIST")
                .help("Attributes of the json documents to show in the results")
                .takes_value(true),
        )
        .get_matches();

    let printer = CursedPrinter::new(matches.value_of("attributes"));
    if let Some(json_file_path) = matches.value_of("index") {
        let file = match File::open(json_file_path) {
            Ok(file) => file,
            Err(err) => {
                error!("Error opening json file: {:?}", err);
                return;
            }
        };
        let mut reader = BufReader::new(file);
        let documents: Vec<Value> = match serde_json::from_reader(&mut reader) {
            Ok(documents) => documents,
            Err(err) => {
                error!("Error deserializing json file: {:?}", err);
                return;
            }
        };
        Index::create_index(documents);
        return;
    }
    match Index::open() {
        Ok(index) => {
            if let Some(query) = matches.value_of("no-curses") {
                single_search(index, query);
            } else {
                ncur(index, printer);
            }
        }
        Err(err) => {
            error!("Error opening index: {:?}", err);
        }
    }
}

fn single_search(mut index: Index, query: &str) {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let search_results = index.search(query, 10);
    let mut output_string = String::new();
    let mut result_counter = 0;
    'outer: for (_typo, results) in search_results.iter() {
        for (_distance, id) in results {
            let document = index.fetch_document(*id).unwrap();
            output_string += &format!("{:#}", document);

            result_counter += 1;
            if result_counter >= 10 {
                break 'outer;
            }
        }
    }
    println!("{}", output_string);
}

fn ncur(mut index: Index, printer: CursedPrinter) {
    initscr();
    /* Allow for extended keyboard (like F1). */
    keypad(stdscr(), true);
    noecho();
    clear();
    attron(A_REVERSE());
    addstr(&&format!("Exit: F1 ||\n",));
    attroff(A_REVERSE());
    attron(A_BOLD());
    addstr("Search: ");
    attroff(A_BOLD());
    refresh();
    let mut text_query = String::new();
    loop {
        let ch = getch();
        match ch {
            KEY_F1 => {
                endwin();
                return;
            }
            KEY_ENTER => {}
            KEY_BACKSPACE | KEY_DC => {
                text_query.pop();
            }
            KEY_DL => {
                text_query.clear();
            }
            _ => {
                let c = char::from_u32(ch as u32).expect("Invalid char");
                text_query.push(c);
            }
        }
        let start_instant = Instant::now();
        let search_results = index.search(&text_query, 10);
        let elapsed = start_instant.elapsed().as_micros();
        let legend = format!(
            "Exit: F1 or Ctrl+C || Found {} results in {}us",
            search_results
                .iter()
                .map(|(_, ids)| ids.len())
                .sum::<usize>(),
            elapsed
        );

        clear();
        attron(A_REVERSE());
        addstr(&legend);
        attroff(A_REVERSE());
        attron(A_BOLD());
        addstr("\nSearch: ");
        attroff(A_BOLD());
        addstr(&text_query);
        addstr("\n");

        let mut result_counter = 0;
        'outer: for (_typo, results) in search_results.iter() {
            for (_distance, id) in results {
                let document = index.fetch_document(*id).unwrap();
                addstr("\n");
                printer.print(document);

                result_counter += 1;
                if result_counter >= 10 {
                    break 'outer;
                }
            }
        }

        refresh();
    }
}
