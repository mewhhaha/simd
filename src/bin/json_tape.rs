use serde_json::{Map, Number, Value};
use std::io::{self, Read};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JsonTag {
    Null,
    Bool,
    Num,
    Str,
    Arr,
    Obj,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ObjEntry {
    key_off: u32,
    key_len: u32,
    value_root: u32,
}

#[derive(Clone, Debug, Default)]
struct JsonTape {
    tags: Vec<JsonTag>,
    bool_v: Vec<bool>,
    num_off: Vec<u32>,
    num_len: Vec<u32>,
    str_off: Vec<u32>,
    str_len: Vec<u32>,
    first_child: Vec<u32>,
    child_count: Vec<u32>,
    first_entry: Vec<u32>,
    entry_count: Vec<u32>,
    subtree_end: Vec<u32>,
    obj_entries: Vec<ObjEntry>,
    bytes: Vec<u8>,
}

impl JsonTape {
    fn encode(root: &Value) -> (Self, u32) {
        let mut tape = Self::default();
        let root_id = tape.push(root);
        (tape, root_id)
    }

    fn decode(&self, root: u32) -> Value {
        match self.tag(root) {
            JsonTag::Null => Value::Null,
            JsonTag::Bool => Value::Bool(self.bool_v[root as usize]),
            JsonTag::Num => {
                let text = self.num_at(root);
                let number: Number = serde_json::from_str(text)
                    .unwrap_or_else(|_| panic!("invalid stored JSON number: {text}"));
                Value::Number(number)
            }
            JsonTag::Str => Value::String(self.str_at(root).to_string()),
            JsonTag::Arr => {
                Value::Array(self.children(root).into_iter().map(|id| self.decode(id)).collect())
            }
            JsonTag::Obj => {
                let mut object = Map::new();
                for entry in self.object_entries(root) {
                    object.insert(self.key_at(entry).to_string(), self.decode(entry.value_root));
                }
                Value::Object(object)
            }
        }
    }

    fn push(&mut self, value: &Value) -> u32 {
        let id = self.reserve_node();
        match value {
            Value::Null => {
                self.tags[id as usize] = JsonTag::Null;
            }
            Value::Bool(v) => {
                self.tags[id as usize] = JsonTag::Bool;
                self.bool_v[id as usize] = *v;
            }
            Value::Number(v) => {
                self.tags[id as usize] = JsonTag::Num;
                let text = v.to_string();
                let off = self.bytes.len() as u32;
                self.bytes.extend_from_slice(text.as_bytes());
                self.num_off[id as usize] = off;
                self.num_len[id as usize] = text.len() as u32;
            }
            Value::String(v) => {
                self.tags[id as usize] = JsonTag::Str;
                let off = self.bytes.len() as u32;
                self.bytes.extend_from_slice(v.as_bytes());
                self.str_off[id as usize] = off;
                self.str_len[id as usize] = v.len() as u32;
            }
            Value::Array(items) => {
                self.tags[id as usize] = JsonTag::Arr;
                self.child_count[id as usize] = items.len() as u32;
                if !items.is_empty() {
                    self.first_child[id as usize] = self.tags.len() as u32;
                }
                for item in items {
                    self.push(item);
                }
            }
            Value::Object(fields) => {
                self.tags[id as usize] = JsonTag::Obj;
                self.first_entry[id as usize] = self.obj_entries.len() as u32;
                self.entry_count[id as usize] = fields.len() as u32;
                for (key, value) in fields {
                    let key_off = self.bytes.len() as u32;
                    self.bytes.extend_from_slice(key.as_bytes());
                    let value_root = self.tags.len() as u32;
                    self.push(value);
                    self.obj_entries.push(ObjEntry {
                        key_off,
                        key_len: key.len() as u32,
                        value_root,
                    });
                }
            }
        }
        self.subtree_end[id as usize] = self.tags.len() as u32;
        id
    }

    fn reserve_node(&mut self) -> u32 {
        let id = self.tags.len() as u32;
        self.tags.push(JsonTag::Null);
        self.bool_v.push(false);
        self.num_off.push(0);
        self.num_len.push(0);
        self.str_off.push(0);
        self.str_len.push(0);
        self.first_child.push(0);
        self.child_count.push(0);
        self.first_entry.push(0);
        self.entry_count.push(0);
        self.subtree_end.push(0);
        id
    }

    fn tag(&self, id: u32) -> JsonTag {
        self.tags[id as usize]
    }

    fn skip_subtree(&self, id: u32) -> u32 {
        self.subtree_end[id as usize]
    }

    fn children(&self, id: u32) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.child_count[id as usize] as usize);
        let mut cur = self.first_child[id as usize];
        for _ in 0..self.child_count[id as usize] {
            out.push(cur);
            cur = self.subtree_end[cur as usize];
        }
        out
    }

    fn object_entries(&self, id: u32) -> &[ObjEntry] {
        let off = self.first_entry[id as usize] as usize;
        let len = self.entry_count[id as usize] as usize;
        &self.obj_entries[off..off + len]
    }

    fn num_at(&self, id: u32) -> &str {
        let off = self.num_off[id as usize] as usize;
        let len = self.num_len[id as usize] as usize;
        std::str::from_utf8(&self.bytes[off..off + len]).expect("utf-8 number payload")
    }

    fn str_at(&self, id: u32) -> &str {
        let off = self.str_off[id as usize] as usize;
        let len = self.str_len[id as usize] as usize;
        std::str::from_utf8(&self.bytes[off..off + len]).expect("utf-8 string payload")
    }

    fn key_at<'a>(&'a self, entry: &ObjEntry) -> &'a str {
        let off = entry.key_off as usize;
        let len = entry.key_len as usize;
        std::str::from_utf8(&self.bytes[off..off + len]).expect("utf-8 key payload")
    }

    fn dump(&self) -> String {
        let mut out = String::new();
        for id in 0..self.tags.len() {
            let id = id as u32;
            match self.tag(id) {
                JsonTag::Null => {
                    out.push_str(&format!("{id}: Null end={}\n", self.skip_subtree(id)));
                }
                JsonTag::Bool => {
                    out.push_str(&format!(
                        "{id}: Bool({}) end={}\n",
                        self.bool_v[id as usize],
                        self.skip_subtree(id)
                    ));
                }
                JsonTag::Num => {
                    out.push_str(&format!(
                        "{id}: Num({}) end={}\n",
                        self.num_at(id),
                        self.skip_subtree(id)
                    ));
                }
                JsonTag::Str => {
                    out.push_str(&format!(
                        "{id}: Str({:?}) end={}\n",
                        self.str_at(id),
                        self.skip_subtree(id)
                    ));
                }
                JsonTag::Arr => {
                    out.push_str(&format!(
                        "{id}: Arr children={:?} end={}\n",
                        self.children(id),
                        self.skip_subtree(id)
                    ));
                }
                JsonTag::Obj => {
                    let pairs = self
                        .object_entries(id)
                        .iter()
                        .map(|entry| format!("{}=>{}", self.key_at(entry), entry.value_root))
                        .collect::<Vec<_>>()
                        .join(", ");
                    out.push_str(&format!(
                        "{id}: Obj {{{pairs}}} end={}\n",
                        self.skip_subtree(id)
                    ));
                }
            }
        }
        out
    }
}

fn read_input() -> Result<String, String> {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        let mut input = String::new();
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|error| format!("failed to read stdin: {error}"))?;
        if input.trim().is_empty() {
            Err("usage: cargo run --bin json_tape -- '<json>'  (or pipe JSON on stdin)".to_string())
        } else {
            Ok(input)
        }
    } else {
        Ok(args.drain(..).collect::<Vec<_>>().join(" "))
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let input = read_input()?;
    let value: Value = serde_json::from_str(&input)
        .map_err(|error| format!("failed to parse JSON input: {error}"))?;
    let (tape, root) = JsonTape::encode(&value);
    println!("root={root}");
    print!("{}", tape.dump());
    let roundtrip = tape.decode(root);
    if roundtrip != value {
        return Err(format!(
            "roundtrip mismatch\noriginal: {}\nroundtrip: {}",
            value, roundtrip
        ));
    }
    println!("roundtrip=ok");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_json() -> Value {
        serde_json::json!([
            1,
            {"x": [true, null], "msg": "hi"},
            "tail"
        ])
    }

    #[test]
    fn encodes_preorder_tape_with_jump_metadata() {
        let (tape, root) = JsonTape::encode(&sample_json());
        assert_eq!(root, 0);
        assert_eq!(tape.tag(root), JsonTag::Arr);
        assert_eq!(tape.children(root), vec![1, 2, 7]);

        assert_eq!(tape.tag(1), JsonTag::Num);
        assert_eq!(tape.num_at(1), "1");

        assert_eq!(tape.tag(2), JsonTag::Obj);
        let entries = tape.object_entries(2);
        assert_eq!(entries.len(), 2);
        assert_eq!(tape.key_at(&entries[0]), "msg");
        assert_eq!(tape.key_at(&entries[1]), "x");
        assert_eq!(tape.tag(entries[0].value_root), JsonTag::Str);
        assert_eq!(tape.tag(entries[1].value_root), JsonTag::Arr);
        assert_eq!(tape.children(entries[1].value_root), vec![5, 6]);

        assert_eq!(tape.tag(7), JsonTag::Str);
        assert_eq!(tape.str_at(7), "tail");
    }

    #[test]
    fn skip_subtree_jumps_to_next_sibling_root() {
        let (tape, root) = JsonTape::encode(&sample_json());
        let object = tape.children(root)[1];
        assert_eq!(object, 2);
        assert_eq!(tape.skip_subtree(object), 7);
        assert_eq!(tape.tag(tape.skip_subtree(object)), JsonTag::Str);
    }

    #[test]
    fn roundtrips_through_tape() {
        let value = sample_json();
        let (tape, root) = JsonTape::encode(&value);
        assert_eq!(tape.decode(root), value);
    }
}
