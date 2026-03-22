#[derive(Clone, Debug, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<JsonValue>),
    Obj(Vec<(String, JsonValue)>),
}

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
    num_v: Vec<f64>,
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
    fn encode(root: &JsonValue) -> (Self, u32) {
        let mut tape = Self::default();
        let root_id = tape.push(root);
        (tape, root_id)
    }

    fn push(&mut self, value: &JsonValue) -> u32 {
        let id = self.reserve_node();
        match value {
            JsonValue::Null => {
                self.tags[id as usize] = JsonTag::Null;
            }
            JsonValue::Bool(v) => {
                self.tags[id as usize] = JsonTag::Bool;
                self.bool_v[id as usize] = *v;
            }
            JsonValue::Num(v) => {
                self.tags[id as usize] = JsonTag::Num;
                self.num_v[id as usize] = *v;
            }
            JsonValue::Str(v) => {
                self.tags[id as usize] = JsonTag::Str;
                let off = self.bytes.len() as u32;
                self.bytes.extend_from_slice(v.as_bytes());
                self.str_off[id as usize] = off;
                self.str_len[id as usize] = v.len() as u32;
            }
            JsonValue::Arr(items) => {
                self.tags[id as usize] = JsonTag::Arr;
                self.child_count[id as usize] = items.len() as u32;
                if !items.is_empty() {
                    self.first_child[id as usize] = self.tags.len() as u32;
                }
                for item in items {
                    self.push(item);
                }
            }
            JsonValue::Obj(fields) => {
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
        self.num_v.push(0.0);
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

    fn str_at(&self, id: u32) -> &str {
        let off = self.str_off[id as usize] as usize;
        let len = self.str_len[id as usize] as usize;
        std::str::from_utf8(&self.bytes[off..off + len]).expect("utf-8 payload")
    }

    fn key_at<'a>(&'a self, entry: &ObjEntry) -> &'a str {
        let off = entry.key_off as usize;
        let len = entry.key_len as usize;
        std::str::from_utf8(&self.bytes[off..off + len]).expect("utf-8 key")
    }
}

fn sample_json() -> JsonValue {
    JsonValue::Arr(vec![
        JsonValue::Num(1.0),
        JsonValue::Obj(vec![(
            "x".to_string(),
            JsonValue::Arr(vec![JsonValue::Bool(true), JsonValue::Null]),
        )]),
        JsonValue::Str("hi".to_string()),
    ])
}

fn main() {
    let (tape, root) = JsonTape::encode(&sample_json());
    println!("root={root} tag={:?} children={:?}", tape.tag(root), tape.children(root));
    let object = tape.children(root)[1];
    let entry = &tape.object_entries(object)[0];
    println!(
        "field={} value_root={} value_tag={:?} subtree_end={}",
        tape.key_at(entry),
        entry.value_root,
        tape.tag(entry.value_root),
        tape.skip_subtree(entry.value_root)
    );
    let string_node = tape.children(root)[2];
    println!("string node {} => {:?}", string_node, tape.str_at(string_node));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_preorder_tape_with_jump_metadata() {
        let (tape, root) = JsonTape::encode(&sample_json());
        assert_eq!(root, 0);
        assert_eq!(tape.tag(root), JsonTag::Arr);
        assert_eq!(tape.children(root), vec![1, 2, 6]);

        assert_eq!(tape.tag(1), JsonTag::Num);
        assert_eq!(tape.num_v[1], 1.0);

        assert_eq!(tape.tag(2), JsonTag::Obj);
        let entry = &tape.object_entries(2)[0];
        assert_eq!(tape.key_at(entry), "x");
        assert_eq!(entry.value_root, 3);

        assert_eq!(tape.tag(3), JsonTag::Arr);
        assert_eq!(tape.children(3), vec![4, 5]);
        assert_eq!(tape.tag(4), JsonTag::Bool);
        assert!(tape.bool_v[4]);
        assert_eq!(tape.tag(5), JsonTag::Null);

        assert_eq!(tape.tag(6), JsonTag::Str);
        assert_eq!(tape.str_at(6), "hi");
    }

    #[test]
    fn skip_subtree_jumps_to_next_sibling_root() {
        let (tape, root) = JsonTape::encode(&sample_json());
        let object = tape.children(root)[1];
        assert_eq!(object, 2);
        assert_eq!(tape.skip_subtree(object), 6);
        assert_eq!(tape.tag(tape.skip_subtree(object)), JsonTag::Str);
    }
}
