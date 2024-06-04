use std::cmp::Ordering;
use std::collections::BinaryHeap;

// MonotonicRng generates a sequence of indices indicating which slots
// a symbol maps. Increasing probability of longer strides between indices.
// Index i will be present in the generated sequence with probability 1/(1+i/2),
// for any non-negative i.
struct MonotonicRng {
    seed: u64,
    last_idx: u64,
}

// floor( ( (1+sqrt(5))/2 ) * 2**64 MOD 2**64)
const GOLDEN_GAMMA: u64 = 0x9E3779B97F4A7C15;

impl MonotonicRng {
    // Fast Splittable Pseudorandom Number Generators
    // Steele Jr, Guy L., Doug Lea, and Christine H. Flood.
    // "Fast splittable pseudorandom number generators."
    fn split_mix(&mut self) -> u64 {
        self.seed += GOLDEN_GAMMA;
        let mut z = self.seed;
        // David Stafford's Mix13 for MurmurHash3's 64-bit finalizer
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }

    fn next(&mut self) -> u64 {
        let rand = self.split_mix();
        // Calculate the difference from the current index (self.last_idx) to the next
        // index. See the paper for details. We use the approximated form
        //   diff = (1.5+i)((1-u)^(-1/2)-1)
        // where i is the current index, i.e., last_idx; u is a number uniformly
        // sampled from [0, 1). We apply the following optimization. Notice that
        // our u actually comes from sampling a random u64 r, and then dividing
        // it by maxUint64, i.e., 1<<64. So we can replace (1-u)^(-1/2) with
        //   1<<32 / sqrt(r).
        self.last_idx += ((self.last_idx as f64 + 1.5)
            * ((1 << 32) as f64 / ((rand + 1) as f64).sqrt() - 1.0))
            .ceil() as u64;
        self.last_idx
    }
}

// Symbol is the interface that source symbols should implement. It specifies a
// Boolean group, where T (or its subset) is the underlying set, and ^ is the
// group operation. It should satisfy the following properties:
//  1. For all a, b, c in the group, (a ^ b) ^ c = a ^ (b ^ c).
//  2. Let e be the default value of T. For every a in the group, e ^ a = a
//     and a ^ e = a.
//  3. For every a in the group, a ^ a = e.
pub trait Symbol: Default + Clone {
    // XOR returns t ^ t2, where t is the method receiver. XOR is allowed to
    // modify the method receiver. Although the method is called XOR (because
    // the bitwise exclusive-or operation is a valid group operation for groups
    // of fixed-length bit strings), it can implement any operation that
    // satisfy the aforementioned properties.
    fn xor(&mut self, t2: &Self);

    // Hash returns the hash of the method receiver. It must not modify the
    // method receiver. It must not be homomorphic over the group operation.
    // That is, the probability that
    //   (a ^ b).Hash() == a.Hash() ^ b.Hash()
    // must be negligible. Here, ^ is the group operation on the left-hand
    // side, and bitwise exclusive-or on the right side.
    fn hash(&self) -> u64;
}

// HashedSymbol is the bundle of a symbol and its hash.
#[derive(Clone, Copy, Default)]
pub struct HashedSymbol<T: Symbol> {
    pub symbol: T,
    pub hash: u64,
}

impl<T: Symbol> HashedSymbol<T> {
    fn new(symbol: T) -> Self {
        let hash = symbol.hash();
        HashedSymbol { symbol, hash }
    }

    fn xor(&mut self, t2: &Self) {
        self.symbol.xor(&t2.symbol);
        self.hash ^= t2.hash;
    }
}

// CodedSymbol is a coded symbol produced by a Rateless IBLT encoder.
#[derive(Clone, Copy, Default)]
pub struct CodedSymbol<T: Symbol> {
    pub inner: HashedSymbol<T>,
    pub count: i64,
}

#[repr(i64)]
#[derive(Clone, Copy)]
enum ApplySign {
    Add = 1,
    Remove = -1,
}

pub const ADD: i64 = 1;
pub const REMOVE: i64 = -1;

impl<T: Symbol> CodedSymbol<T> {
    // XORs the symbol and updates the count depending on the sign
    // of the apply (adding or removing the symbol).
    pub fn apply(&mut self, hs: &HashedSymbol<T>, sign: ApplySign) {
        self.inner.xor(hs);
        self.count += sign as i64;
    }
}

// symbolMapping is a mapping from a source symbol to a coded symbol. The
// symbols are identified by their indices in codingWindow.
struct SymbolMapping {
    source_idx: usize,
    coded_idx: usize,
}

impl Eq for SymbolMapping {}
impl PartialEq for SymbolMapping {
    fn eq(&self, other: &Self) -> bool {
        self.coded_idx == other.coded_idx
    }
}

impl PartialOrd for SymbolMapping {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.coded_idx.partial_cmp(&other.coded_idx)
    }
}

impl Ord for SymbolMapping {
    fn cmp(&self, other: &Self) -> Ordering {
        self.coded_idx.cmp(&other.coded_idx)
    }
}

// codingWindow is a collection of source symbols and their mappings to coded symbols.
struct CodingWindow<T: Symbol> {
    symbols: Vec<HashedSymbol<T>>,    // source symbols
    mappings: Vec<MonotonicRng>,      // mapping generators of the source symbols
    queue: BinaryHeap<SymbolMapping>, // priority queue of source symbols by the next coded symbols they are mapped to
    next_idx: usize,                  // index of the next coded symbol to be generated
}

impl<T: Symbol> CodingWindow<T> {
    // add_symbol inserts a symbol to the codingWindow.
    fn add_symbol(&mut self, symbol: T) {
        self.add_hashed_symbol(HashedSymbol::new(symbol));
    }

    // add_hashed_symbol inserts a HashedSymbol to the codingWindow.
    fn add_hashed_symbol(&mut self, hs: HashedSymbol<T>) {
        let seed = hs.hash;
        self.add_hashed_symbol_with_mapping(hs, MonotonicRng { seed, last_idx: 0 });
    }

    // add_hashed_symbol_with_mapping inserts a HashedSymbol and the current state of its mapping generator to the codingWindow.
    fn add_hashed_symbol_with_mapping(&mut self, hs: HashedSymbol<T>, rng: MonotonicRng) {
        self.queue.push(SymbolMapping {
            source_idx: self.symbols.len() - 1,
            coded_idx: rng.last_idx as usize,
        });
        self.symbols.push(hs);
        self.mappings.push(rng);
    }

    // apply_window maps the source symbols to the next coded symbol they should be
    // mapped to, given as cw. The parameter direction controls how the counter
    // of cw should be modified.
    fn apply_window(&mut self, mut cw: CodedSymbol<T>, sign: ApplySign) -> CodedSymbol<T> {
        if self.queue.is_empty() {
            self.next_idx += 1;
            return cw;
        }
        while let Some(mut next) = self.queue.peek_mut() {
            if next.coded_idx == self.next_idx {
                cw.apply(&self.symbols[next.source_idx], sign);
                // generate the next mapping
                let next_map = self.mappings[next.source_idx].next();
                next.coded_idx = next_map as usize;
            }
        }
        self.next_idx += 1;
        cw
    }

    // reset clears a codingWindow.
    fn reset(&mut self) {
        self.symbols.clear();
        self.mappings.clear();
        self.queue.clear();
        self.next_idx = 0;
    }
}

// Encoder is an incremental encoder of Rateless IBLT.
struct Encoder<T: Symbol>(CodingWindow<T>);

impl<T: Symbol> Encoder<T> {
    // AddSymbol adds a symbol to the encoder.
    fn add_symbol(&mut self, s: T) {
        self.0.add_symbol(s);
    }

    // AddHashedSymbol adds a HashedSymbol to the encoder.
    fn add_hashed_symbol(&mut self, hs: HashedSymbol<T>) {
        self.0.add_hashed_symbol(hs);
    }

    // ProduceNextCodedSymbol returns the next coded symbol the encoder produces.
    fn produce_next_coded_symbol(&mut self) -> CodedSymbol<T> {
        // TODO: Using Default to initialize a zero CodedSymbol is not ideal.
        self.0.apply_window(CodedSymbol::default(), ApplySign::Add)
    }

    // Reset clears the encoder.
    fn reset(&mut self) {
        self.0.reset();
    }
}

// Decoder is an incremental decoder of Rateless IBLT.
struct Decoder<T: Symbol> {
    // coded symbols received so far
    cs: Vec<CodedSymbol<T>>,
    // set of source symbols that are exclusive to the decoder
    local: CodingWindow<T>,
    // set of source symbols that the decoder initially has
    window: CodingWindow<T>,
    // set of source symbols that are exclusive to the encoder
    remote: CodingWindow<T>,
    // indices of coded symbols that can be decoded, i.e., degree equal to -1
    // or 1 and sum of hash equal to hash of sum, or degree equal to 0 and sum
    // of hash equal to 0
    decodable: Vec<usize>,
    // number of coded symbols that are decoded
    decoded: usize,
}

impl<T: Symbol> Decoder<T> {
    fn decoded(&self) -> bool {
        self.decoded == self.cs.len()
    }

    fn local(&self) -> &[HashedSymbol<T>] {
        &self.local.symbols
    }

    fn remote(&self) -> &[HashedSymbol<T>] {
        &self.remote.symbols
    }

    fn add_symbol(&mut self, symbol: T) {
        self.add_hashed_symbol(HashedSymbol::new(symbol));
    }

    fn add_hashed_symbol(&mut self, hs: HashedSymbol<T>) {
        self.window.add_hashed_symbol(hs);
    }

    fn add_coded_symbol(&mut self, mut cs: CodedSymbol<T>) {
        // scan through decoded symbols to peel off matching ones
        cs = self.window.apply_window(cs, ApplySign::Remove);
        cs = self.remote.apply_window(cs, ApplySign::Remove);
        cs = self.local.apply_window(cs, ApplySign::Add);
        // check if the coded symbol is decodable, and insert into decodable list if so
        if (cs.count == 1 || cs.count == -1) && cs.inner.hash == cs.inner.symbol.hash() {
            self.decodable.push(self.cs.len());
        } else if cs.count == 0 && cs.inner.hash == 0 {
            self.decodable.push(self.cs.len());
        }
        // insert the new coded symbol
        self.cs.push(cs);
    }

    fn apply_new_symbol(&mut self, hs: &HashedSymbol<T>, sign: ApplySign) -> MonotonicRng {
        let mut rng = MonotonicRng {
            seed: hs.hash,
            last_idx: 0,
        };

        while (rng.last_idx as usize) < self.cs.len() {
            let cidx = rng.last_idx as usize;
            self.cs[cidx].apply(&hs, sign);
            if (self.cs[cidx].count == -1 || self.cs[cidx].count == 1)
                && self.cs[cidx].inner.hash == self.cs[cidx].inner.symbol.hash()
            {
                self.decodable.push(cidx);
            }
            rng.next();
        }
        rng
    }

    fn try_decode(&mut self) {
        for didx in 0..self.decodable.len() {
            let cidx = self.decodable[didx];
            let c = &self.cs[cidx];
            match c.count {
                1 => {
                    let mut ns = HashedSymbol::<T>::default();
                    ns.symbol.xor(&c.inner.symbol);
                    ns.hash = c.inner.hash;
                    let m = self.apply_new_symbol(&ns, ApplySign::Remove);
                    self.remote.add_hashed_symbol_with_mapping(ns, m);
                    self.decoded += 1;
                }
                -1 => {
                    let mut ns = HashedSymbol::<T>::default();
                    ns.symbol.xor(&c.inner.symbol);
                    ns.hash = c.inner.hash;
                    let m = self.apply_new_symbol(&ns, ApplySign::Add);
                    self.local.add_hashed_symbol_with_mapping(ns, m);
                    self.decoded += 1;
                }
                0 => {
                    self.decoded += 1;
                }
                _ => panic!("invalid degree for decodable coded symbol"),
            }
        }
        self.decodable.clear();
    }

    fn reset(&mut self) {
        self.cs.clear();
        self.decodable.clear();
        self.local.reset();
        self.remote.reset();
        self.window.reset();
        self.decoded = 0;
    }
}
