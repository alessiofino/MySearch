use std::{
    cmp,
    collections::{HashMap, HashSet},
    error::Error,
    fs::{self, File},
    io::BufWriter,
    time::Instant,
};

use fnv::FnvHashMap;
use fst::{IntoStreamer, Set, SetBuilder, Streamer};
use heed::{
    types::{CowSlice, CowType, UnalignedSlice},
    Database, Env, EnvOpenOptions, RoTxn,
};
use itertools::Itertools;
use levenshtein_automata::{LevenshteinAutomatonBuilder as LevBuilder, DFA};
use log::debug;
use lru::LruCache;
use meilisearch_tokenizer::{Analyzer, AnalyzerConfig};
use memmap::{Mmap, MmapOptions};
use once_cell::sync::OnceCell;
use sdset::{duo::OpBuilder, SetBuf, SetOperation};
use serde_json::Value;

static LEVDIST0: OnceCell<LevBuilder> = OnceCell::new();
static LEVDIST1: OnceCell<LevBuilder> = OnceCell::new();
static LEVDIST2: OnceCell<LevBuilder> = OnceCell::new();

type WordKey = UnalignedSlice<u8>;
type PosKey = UnalignedSlice<u8>;
type DocId = u32;
type WordPos = u16;
type PrefixKey = UnalignedSlice<u8>;
type SerializedDoc = UnalignedSlice<u8>;
type Typo = u8;
type WordDistance = u16;

pub struct Index {
    env: Env,
    /*Stores the list of DocId of the documents containing
    a specific word, the WordKey is the byte representation
    of the word*/
    rev_index_kv: Database<WordKey, CowSlice<DocId>>,
    /*Stores the positions of a word inside a document,
    the WordKey is the concatenation of the bytes of the
    word and the DocId*/
    word_positions_kv: Database<PosKey, CowSlice<WordPos>>,
    /*Stores the list of DocIds of the documents containing
    a specific prefix, the PrefixKey is the byte representation
    of the prefix, the prefix is a sequence of 1-2 chars making
    the PrefixKey 4-8 bytes long*/
    prefix_kv: Database<PrefixKey, CowSlice<DocId>>,
    /*Stores the serialized documents, the format used is json*/
    documents_kv: Database<CowType<DocId>, SerializedDoc>,
    /*Contains all the unique words in the dataset of documents */
    word_set: Set<Mmap>,
    analyzer: Analyzer<Vec<u8>>,
    /*Used to cache expensive searches for words with a typo level
    of 2, it helps greatly reduce the processing times when performing
    a search as you type */
    lru_cache: LruCache<String, Vec<(Typo, String)>>,
}

impl Index {
    pub fn open() -> Result<Self, Box<dyn Error>> {
        fs::create_dir_all("db/db.mdb").unwrap();
        let env = EnvOpenOptions::new()
            .map_size(100 * 1024 * 1024 * 1024)
            .max_dbs(10)
            .open("db/db.mdb")?;

        let rev_index_kv: Database<WordKey, CowSlice<DocId>> = env
            .open_database(Some("wordid_to_doc"))?
            .ok_or("Error opening rev_index db")?;

        let word_positions_kv: Database<PosKey, CowSlice<WordPos>> = env
            .open_database(Some("positions"))?
            .ok_or("Error opening word_positions_db")?;

        let prefix_kv: Database<PrefixKey, CowSlice<DocId>> = env
            .open_database(Some("prefix"))?
            .ok_or("Error opening prefix_db")?;

        let documents_kv: Database<CowType<DocId>, SerializedDoc> = env
            .open_database(Some("documents"))?
            .ok_or("Error opening documents_db")?;

        let set_file = File::open("db/words.fst")?;
        let mmap = unsafe { MmapOptions::new().map(&set_file)? };
        let word_set = Set::new(mmap)?;

        let stop_words = Set::default();
        let analyzer = Analyzer::new(AnalyzerConfig::default_with_stopwords(stop_words));

        Ok(Self {
            env,
            rev_index_kv,
            word_positions_kv,
            prefix_kv,
            documents_kv,
            word_set,
            analyzer,
            lru_cache: LruCache::new(100),
        })
    }

    pub fn create_index(documents: Vec<Value>) {
        //I consider only text fields for now
        fn extract_words(value: &Value, analyzer: &Analyzer<Vec<u8>>) -> Vec<String> {
            let result = match value {
                Value::String(text) => {
                    let analyzed_text = analyzer.analyze(&text);
                    analyzed_text
                        .tokens()
                        .filter(|t| t.is_word())
                        .map(|t| t.word.to_string())
                        .collect()
                }
                Value::Array(vec) => vec
                    .iter()
                    .flat_map(|value| extract_words(value, analyzer))
                    .collect(),
                Value::Object(map) => map
                    .iter()
                    .flat_map(|(_, value)| extract_words(value, analyzer))
                    .collect(),
                _ => {
                    vec![]
                }
            };
            result
        }

        fs::create_dir_all("db/db.mdb").unwrap();
        let env = EnvOpenOptions::new()
            .map_size(100 * 1024 * 1024 * 1024)
            .max_dbs(10)
            .open("db/db.mdb")
            .unwrap();

        let rev_index_kv: Database<WordKey, CowSlice<DocId>> =
            if let Ok(Some(db)) = env.open_database(Some("wordid_to_doc")) {
                db
            } else {
                env.create_database(Some("wordid_to_doc")).unwrap()
            };
        let word_positions_kv: Database<PosKey, CowSlice<WordPos>> =
            if let Ok(Some(db)) = env.open_database(Some("positions")) {
                db
            } else {
                env.create_database(Some("positions")).unwrap()
            };
        let prefix_kv: Database<PrefixKey, CowSlice<DocId>> =
            if let Ok(Some(db)) = env.open_database(Some("prefix")) {
                db
            } else {
                env.create_database(Some("prefix")).unwrap()
            };

        let documents_kv: Database<CowType<DocId>, SerializedDoc> =
            if let Ok(Some(db)) = env.open_database(Some("documents")) {
                db
            } else {
                env.create_database(Some("documents")).unwrap()
            };

        let mut wtxn = env.write_txn().unwrap();

        rev_index_kv.clear(&mut wtxn).unwrap();
        word_positions_kv.clear(&mut wtxn).unwrap();
        prefix_kv.clear(&mut wtxn).unwrap();
        documents_kv.clear(&mut wtxn).unwrap();

        let buff_writer = BufWriter::new(File::create("db/words.fst").unwrap());
        let mut set_builder = SetBuilder::new(buff_writer).unwrap();

        let stop_words = Set::default();
        let analyzer = Analyzer::new(AnalyzerConfig::default_with_stopwords(stop_words));

        let mut words_to_doc_map: HashMap<String, Vec<DocId>> = HashMap::new();
        let mut key_buffer: Vec<u8> = Vec::new();

        for (doc_id, doc) in documents.iter().enumerate() {
            let doc_id = doc_id as DocId;

            let doc_words = extract_words(doc, &analyzer);
            let doc_words_positions: HashMap<String, Vec<WordPos>> = doc_words
                .into_iter()
                .enumerate()
                .map(|(pos, word)| (word, pos as WordPos))
                .into_group_map();

            for (word, positions) in doc_words_positions {
                key_buffer.clear();
                key_buffer.extend_from_slice(word.as_bytes());
                key_buffer.extend_from_slice(&doc_id.to_ne_bytes());
                word_positions_kv
                    .put(&mut wtxn, &key_buffer, &positions)
                    .unwrap();
                words_to_doc_map
                    .entry(word)
                    .or_insert(Vec::new())
                    .push(doc_id);
            }
        }

        for (word, doc_ids) in words_to_doc_map
            .into_iter()
            .sorted_by(|(w1, _), (w2, _)| w1.cmp(w2))
        {
            rev_index_kv
                .put(&mut wtxn, &word.as_bytes(), &doc_ids)
                .unwrap();
            set_builder.insert(word).unwrap();
        }
        set_builder.finish().unwrap();

        let mut buffer = Vec::new();
        for (doc_id, doc) in documents.into_iter().enumerate() {
            let doc_id = doc_id as DocId;
            buffer.clear();
            serde_json::to_writer(&mut buffer, &doc).unwrap();
            documents_kv.put(&mut wtxn, &doc_id, &buffer).unwrap();
        }

        wtxn.commit().unwrap();

        let set_file = File::open("db/words.fst").unwrap();
        let mmap = unsafe { MmapOptions::new().map(&set_file).unwrap() };
        let word_set = Set::new(mmap).unwrap();

        let mut prefix_map: HashMap<Vec<char>, HashSet<DocId>> = HashMap::new();
        let rtxn = env.read_txn().unwrap();
        for prefix_len in 1..=2 {
            let mut stream = word_set.into_stream();
            while let Some(word) = stream.next() {
                let word_str = std::str::from_utf8(word).unwrap();
                if word_str.chars().count() < prefix_len {
                    continue;
                }
                let doc_ids = rev_index_kv.get(&rtxn, word).unwrap().unwrap();
                let prefix_bytes: Vec<char> = word_str.chars().take(prefix_len).collect();

                prefix_map
                    .entry(prefix_bytes)
                    .or_insert(HashSet::new())
                    .extend(doc_ids.iter());
            }
        }
        rtxn.commit().unwrap();

        let mut wtxn = env.write_txn().unwrap();

        for (prefix_chars, ids) in prefix_map {
            let ids: Vec<DocId> = ids.into_iter().sorted().collect();
            let bytes: Vec<u8> = prefix_chars
                .into_iter()
                .flat_map(|c| (c as u32).to_ne_bytes().to_vec())
                .collect();
            prefix_kv.put(&mut wtxn, &bytes, &ids).unwrap();
        }
        wtxn.commit().unwrap();
    }

    /* This function is optimized for searches containing a single word,
    in this cases I can stop immedialty the search if I found enugh results
    by querying directly the reverse index for the search term */
    fn search_prefix_optimized(
        &self,
        rtxn: &RoTxn,
        prefix: &String,
        len: usize,
    ) -> FnvHashMap<u8, SetBuf<u32>> {
        if let Some(doc_ids) = self.rev_index_kv.get(rtxn, &prefix.as_bytes()).unwrap() {
            if doc_ids.len() >= len {
                let mut result: FnvHashMap<Typo, SetBuf<DocId>> = FnvHashMap::default();
                let setbuf: SetBuf<DocId> = SetBuf::new(doc_ids.to_vec()).unwrap();
                result.insert(0, setbuf);
                return result;
            }
        }
        self.search_prefix(rtxn, prefix)
    }

    fn search_prefix(&self, rtxn: &RoTxn, prefix: &String) -> FnvHashMap<Typo, SetBuf<DocId>> {
        let dfa = build_dfa(&prefix, PrefixSetting::Prefix, typo_level(&prefix));
        let first_byte = prefix.as_bytes()[0];
        let mut match_stream = if first_byte == u8::max_value() {
            self.word_set
                .search_with_state(&dfa)
                .ge(&[first_byte])
                .into_stream()
        } else {
            self.word_set
                .search_with_state(&dfa)
                .ge(&[first_byte])
                .lt(&[first_byte + 1])
                .into_stream()
        };
        let mut result: FnvHashMap<Typo, SetBuf<DocId>> = FnvHashMap::default();
        while let Some((word, dfa_final_state)) = match_stream.next() {
            let distance = dfa.distance(dfa_final_state).to_u8();
            let doc_ids = self.rev_index_kv.get(&rtxn, &word).unwrap().unwrap();
            let id_set = sdset::Set::new(&doc_ids).unwrap();
            match result.get_mut(&distance) {
                Some(set) => {
                    *set = OpBuilder::new(set, &id_set).union().into_set_buf();
                }
                None => {
                    result.insert(distance, id_set.to_set_buf());
                }
            }
        }

        result
    }

    /*Given a vector of search terms retrives all the words that match each term using the
    typo distance determined by the lenght of each search terms, the returned value is a vector containing
    an entry for each search term, the entry is a vector of all the words matching the term with the associated
    typo distance to the term */
    fn search_strings(&mut self, words: Vec<String>) -> Vec<Vec<(Typo, String)>> {
        let mut result: Vec<Vec<(Typo, String)>> = Vec::with_capacity(words.len());
        let len = words.len();
        for word in words {
            /*If the typo level is 2 or greater I check in the LRU cache for the word, if
            it's there I don't need to build the Levenshtein automata and perform the
            search which is quite expensive for words that allows for 2 typos */
            if typo_level(&word) >= 2 {
                if let Some(cached_results) = self.lru_cache.get(&word) {
                    result.push(cached_results.clone());
                    continue;
                }
            }
            let dfa = build_dfa(&word, PrefixSetting::NoPrefix, typo_level(&word));
            let first_byte = word.as_bytes()[0];
            let mut match_stream = if first_byte == u8::max_value() {
                self.word_set
                    .search_with_state(&dfa)
                    .ge(&[first_byte])
                    .into_stream()
            } else {
                self.word_set
                    .search_with_state(&dfa)
                    .ge(&[first_byte])
                    .lt(&[first_byte + 1])
                    .into_stream()
            };
            let mut word_matches: Vec<(Typo, String)> = Vec::new();
            while let Some((word, dfa_final_state)) = match_stream.next() {
                let distance = dfa.distance(dfa_final_state).to_u8();
                word_matches.push((distance, String::from_utf8(word.to_vec()).unwrap()));
            }
            /*If I don't have results I can stop immedialty and return an
            empty vector since the intersection of all results is going to be empty
            anyway */
            if word_matches.is_empty() {
                return vec![];
            }

            if typo_level(&word) >= 2 {
                self.lru_cache.put(word, word_matches.clone());
            }
            result.push(word_matches);
        }
        assert_eq!(len, result.len());
        return result;
    }

    /*Given the vector of word matches for each word in the search query finds all the ids of the documents
    containing each search term and perform an intersection to find the documents containing all the search terms,
    during the process it keeps track of the cumulated tuypo distance for each document.
    The returned value is an HashMap with key the typo distance and values the sets of documents containing the search terms
    with exactly that total typo distance.
    Example the set of ids with key 0 in the hashmap are all the ids containing all the search terms without any */
    fn find_ids(
        &self,
        mut words: Vec<Vec<(Typo, String)>>,
        prefix_ids: &FnvHashMap<Typo, SetBuf<DocId>>,
        rtxn: &RoTxn,
    ) -> FnvHashMap<Typo, SetBuf<DocId>> {
        let mut result: FnvHashMap<Typo, SetBuf<DocId>> = FnvHashMap::default();

        let word_matches = words.pop().unwrap();
        for (typo, word) in word_matches {
            let ids = self
                .rev_index_kv
                .get(&rtxn, &word.as_bytes())
                .unwrap()
                .unwrap();
            let id_set = sdset::Set::new(&ids).unwrap();
            match result.get_mut(&typo) {
                Some(setbuf) => {
                    *setbuf = OpBuilder::new(setbuf, &id_set).union().into_set_buf();
                }
                None => {
                    result.insert(typo, id_set.to_set_buf());
                }
            }
        }
        if words.is_empty() {
            let mut res: FnvHashMap<Typo, SetBuf<DocId>> = FnvHashMap::default();
            for (typo, id_setbuf) in result {
                for (pre_typo, pre_id_setbuf) in prefix_ids {
                    let inter: SetBuf<DocId> = OpBuilder::new(&id_setbuf, pre_id_setbuf)
                        .intersection()
                        .into_set_buf();
                    if !inter.is_empty() {
                        match res.get_mut(&(typo + *pre_typo)) {
                            Some(setbuf) => {
                                *setbuf = OpBuilder::new(setbuf, &inter).union().into_set_buf();
                            }
                            None => {
                                res.insert(typo + *pre_typo, inter);
                            }
                        }
                    }
                }
            }
            return res;
        } else {
            let inner_result: FnvHashMap<Typo, SetBuf<DocId>> =
                self.find_ids(words, prefix_ids, rtxn);

            let mut inner_intersection: FnvHashMap<Typo, SetBuf<DocId>> = FnvHashMap::default();

            for (inn_typo, inn_setbuf) in inner_result {
                for (this_typo, this_setbuf) in &result {
                    let inter: SetBuf<DocId> = OpBuilder::new(&inn_setbuf, this_setbuf)
                        .intersection()
                        .into_set_buf();
                    if !inter.is_empty() {
                        match inner_intersection.get_mut(&(inn_typo + *this_typo)) {
                            Some(setbuf) => {
                                *setbuf = OpBuilder::new(setbuf, &inter).union().into_set_buf();
                            }
                            None => {
                                inner_intersection.insert(inn_typo + *this_typo, inter);
                            }
                        }
                    }
                }
            }
            return inner_intersection;
        }
    }

    fn sort_by_distance(
        &self,
        doc_ids: FnvHashMap<Typo, SetBuf<DocId>>,
        matches: Vec<Vec<String>>,
        len: usize,
        rtxn: &RoTxn,
    ) -> FnvHashMap<Typo, Vec<(WordDistance, DocId)>> {
        #[inline]
        fn calculate_distance(positions: &Vec<Vec<WordPos>>) -> WordDistance {
            let mut distance: WordDistance = 0;
            for (first, second) in positions.iter().tuple_windows() {
                for lhs in first {
                    let d: WordDistance = second
                        .iter()
                        .map(|rhs| {
                            if lhs < rhs {
                                cmp::min(rhs - lhs, 8)
                            } else {
                                cmp::min(lhs - rhs, 8) + 1
                            }
                        })
                        .min()
                        .unwrap();
                    distance += d;
                }
            }
            distance
        }

        let mut results_counter = 0;
        let mut results: FnvHashMap<Typo, Vec<(WordDistance, DocId)>> = FnvHashMap::default();
        let mut doc_word_positions: Vec<Vec<WordPos>> = Vec::with_capacity(matches.len());
        let mut key_buffer = Vec::new();
        for (typo, doc_ids) in doc_ids.into_iter().sorted_by(|(t1, _), (t2, _)| t1.cmp(t2)) {
            let mut doc_distances: Vec<(WordDistance, DocId)> = Vec::with_capacity(doc_ids.len());
            results_counter += doc_ids.len();
            for doc_id in doc_ids {
                doc_word_positions.clear();
                for words in &matches {
                    let mut positions: Vec<WordPos> = Vec::new();
                    for word in words {
                        key_buffer.clear();
                        key_buffer.extend_from_slice(word.as_bytes());
                        key_buffer.extend_from_slice(&doc_id.to_ne_bytes());
                        if let Some(pos) = self.word_positions_kv.get(rtxn, &key_buffer).unwrap() {
                            positions.extend_from_slice(&pos);
                        }
                    }
                    assert!(!positions.is_empty());
                    doc_word_positions.push(positions);
                }
                let distance = calculate_distance(&doc_word_positions);
                doc_distances.push((distance, doc_id));
            }
            doc_distances.sort_unstable_by(|(d1, _), (d2, _)| d1.cmp(d2));
            results.insert(typo, doc_distances);

            if results_counter >= len {
                return results;
            }
        }

        results
    }

    pub fn search(
        &mut self,
        query: &str,
        len: usize,
    ) -> FnvHashMap<Typo, Vec<(WordDistance, DocId)>> {
        let rtxn = self.env.read_txn().unwrap();
        let analyzed_text = self.analyzer.analyze(query);
        let mut search_terms: Vec<String> = analyzed_text
            .tokens()
            .filter(|t| t.is_word())
            .map(|t| t.word.to_string())
            .collect();
        if let Some(prefix_term) = search_terms.pop() {
            let prefix_ids: FnvHashMap<Typo, SetBuf<DocId>> = if prefix_term.chars().count() > 2 {
                if search_terms.is_empty() {
                    let prefix_instant = Instant::now();
                    let tmp = self.search_prefix_optimized(&rtxn, &prefix_term, len);
                    debug!(
                        "Search prefix optimized '{}' completed in {}us",
                        prefix_term,
                        prefix_instant.elapsed().as_micros()
                    );
                    tmp
                } else {
                    let prefix_instant = Instant::now();
                    let tmp = self.search_prefix(&rtxn, &prefix_term);
                    debug!(
                        "Search prefix '{}' completed in {}us",
                        prefix_term,
                        prefix_instant.elapsed().as_micros()
                    );
                    tmp
                }
            } else {
                let prefix_instant = Instant::now();
                let key: Vec<u8> = prefix_term
                    .chars()
                    .flat_map(|c| (c as u32).to_ne_bytes().to_vec())
                    .collect();
                let mut prefix_ids: FnvHashMap<Typo, SetBuf<DocId>> = FnvHashMap::default();
                if let Some(ids) = self.prefix_kv.get(&rtxn, &key).unwrap() {
                    let buf = SetBuf::new(ids.to_vec()).unwrap();
                    prefix_ids.insert(0, buf);
                }
                debug!(
                    "Direct search prefix '{}' completed in {}us",
                    prefix_term,
                    prefix_instant.elapsed().as_micros()
                );
                prefix_ids
            };
            if search_terms.is_empty() {
                prefix_ids
                    .into_iter()
                    .map(|(t, set)| (t, set.into_iter().map(|doc_id| (0, doc_id)).collect()))
                    .collect()
            } else {
                drop(rtxn);
                let match_instant = Instant::now();
                let matches: Vec<Vec<(Typo, String)>> = self.search_strings(search_terms);
                debug!(
                    "Match search  completed in {}us",
                    match_instant.elapsed().as_micros()
                );
                let rtxn = self.env.read_txn().unwrap();
                let fetch_instant = Instant::now();
                let ids: FnvHashMap<Typo, SetBuf<DocId>> =
                    self.find_ids(matches.clone(), &prefix_ids, &rtxn);
                debug!(
                    "Id fetch completed in {}us",
                    fetch_instant.elapsed().as_micros()
                );
                let matches: Vec<Vec<String>> = matches
                    .into_iter()
                    .map(|v| v.into_iter().map(|(_, s)| s).collect())
                    .collect();
                if matches.len() >= 2 {
                    let sorting_instant = Instant::now();
                    let sorted = self.sort_by_distance(ids, matches, len, &rtxn);
                    debug!(
                        "Sorting completed in {}us",
                        sorting_instant.elapsed().as_micros()
                    );
                    sorted
                } else {
                    ids.into_iter()
                        .map(|(t, set)| (t, set.into_iter().map(|doc_id| (0, doc_id)).collect()))
                        .collect()
                }
            }
        } else {
            FnvHashMap::default()
        }
    }

    pub fn fetch_document(&self, doc_id: u32) -> Option<Value> {
        let rtxn = self.env.read_txn().unwrap();
        let bytes: &[u8] = self.documents_kv.get(&rtxn, &doc_id).unwrap()?;
        let value: Value = serde_json::from_slice(bytes).unwrap();
        Some(value)
    }
}

#[inline]
fn build_dfa(query: &str, setting: PrefixSetting, typo: u8) -> DFA {
    use PrefixSetting::{NoPrefix, Prefix};

    let builder = match typo {
        0 => LEVDIST0.get_or_init(|| LevBuilder::new(0, true)),
        1 => LEVDIST1.get_or_init(|| LevBuilder::new(1, true)),
        _ => LEVDIST2.get_or_init(|| LevBuilder::new(2, true)),
    };
    match setting {
        Prefix => builder.build_prefix_dfa(query),
        NoPrefix => builder.build_dfa(query),
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum PrefixSetting {
    Prefix,
    NoPrefix,
}

#[inline]
fn typo_level(term: &String) -> u8 {
    match term.len() {
        (0..=4) => 0,
        (5..=8) => 1,
        _ => 2,
    }
}
