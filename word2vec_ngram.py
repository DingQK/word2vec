import sys
import math
import numpy as np

class Ngram:
    def __init__(self, tokens):
        self.tokens = tokens
        self.count = 0
        self.score = 0.0

    def set_score(self, score):
        self.score = score
    
    def get_string(self):
        return '_'.join(self.tokens)


class Corpus:
    def __init__(self, filename, word_phrase_passes, word_phrase_delta, word_phrase_threshold, word_phrase_filename):
        
        cnt = 0

        with open(filename, 'r') as f:
            lines = f.readlines()

        all_tokens = []

        for line in lines:
            line_tokens = line.split()
            for token in line_tokens:
                token = token.lower()

                if len(token) > 1 and token.isalnum():
                    all_tokens.append(token)
                
                cnt += 1
                if cnt % 10000 == 0:
                    sys.stdout.flush()
                    sys.stdout.write("\rReading corpus: %d" % cnt)
            
        sys.stdout.flush()
        print('\rCorpus readL %d'% cnt)

        self.tokens = all_tokens

        for x in range(1, word_phrase_passes + 1):
            self.build_ngrams(x, word_phrase_delta, word_phrase_threshold, word_phrase_filename)
        
        self.save_to_file(filename)
    
    def build_ngrams(self, x, word_phrase_delta, word_phrase_threshold, word_phrase_filename):

        ngrams = []
        ngram_map = {}

        token_count_map = {}
        for token in self.tokens:
            token_count_map.setdefault(token, 1)
            token_count_map[token] += 1
        
        cnt = 0
        ngram_l = []
        for token in self.tokens:

            if len(ngram_l) == 2:
                ngram_l.pop(0)
            
            ngram_l.append(token)
            ngram_t = tuple(ngram_l)

            if ngram_t not in ngram_map:
                ngram_map[ngram_t] = len(ngrams)
                ngrams.append(Ngram(ngram_t))
            
            ngrams[ngram_map[ngram_t]].count += 1

            cnt += 1
            if cnt % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write("\rBuilding n-grams (%d pass): %d" % (x, cnt))
        
        sys.stdout.flush()
        print('\rn-grams (%d pass) built: %d' % (x, cnt))

        filtered_ngrams_map = {}
        f = open(word_phrase_filename+ ('-%d' % x), 'w')

        cnt = 0
        for ngram in ngrams:
            product = 1
            for word_string in ngram.tokens:
                product *= token_count_map[word_string]
            ngram.set_score((float(ngram.count)-word_phrase_delta)/float(product))

            if ngram.score > word_phrase_threshold:
                filtered_ngrams_map[ngram.get_string()] = ngram
                f.write('%s %d\n' % (ngram.get_string(), ngram.count))
            
            cnt += 1
            if cnt % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write("\rScoring n-grams: %d" % cnt)
            
        sys.stdout.flush()
        print("\rScored n-grams: %d, filtered n-grams: %d" % (cnt, len(filtered_ngrams_map)))
        f.close()

        # Combining all tokens
        all_tokens = []
        cnt = 0

        while cnt < len(self.tokens):

            if cnt+1 < len(self.tokens):
                ngram_l = []
                ngram_l.append(self.tokens[cnt])
                ngram_l.append(self.tokens[cnt+1])
                ngram_string = '_'.join(ngram_l)

                if len(ngram_l) == 2 and (ngram_string in filtered_ngrams_map):
                    ngram = filtered_ngrams_map[ngram_string]
                    all_tokens.append(ngram.get_string())
                    cnt += 2
                else:
                    all_tokens.append(self.tokens[cnt])
                    cnt += 1
            else:
                all_tokens.append(self.tokens[cnt])
                cnt += 1
        
        print('Tokens combined')

        self.tokens = all_tokens
    
    def save_to_file(self, filename):

        cnt = 1

        f = open('preprocessed-' + filename.split('/')[-1], 'w')
        line = ''
        for token in self.tokens:
            if cnt % 20 == 0:
                line += token
                f.write('%s\n' % line)
                line = ''
            else:
                line += token + ' '
            cnt += 1

            if cnt % 10000 == 0:
                sys.stdout.flush()
                sys.stdout.write("\rWriting to preprocessed input file")
            
        sys.stdout.flush()
        print("\rPreprocessed input file written")

        f.close()

    def __getitem__(self, i):
        return self.tokens[i]

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)


class Word:
    def __init__(self, word):
        self.word = word
        self.count = 0


class Vocabulary:
    def __init__(self, corpus, min_count):
        self.words = []
        self.word_map = {}
        self.build_words(corpus)

        self.filter_for_rare_and_common(min_count)
    
    def build_words(self, corpus):
        words = []
        word_map = {}

        cnt = 0
        for token in corpus:
            if token not in word_map:
                word_map[token] = len(words)
                words.append(Word(token))
            words[word_map[token]].count += 1
        
        cnt += 1
        if cnt % 10000 == 0:
            sys.stdout.flush()
            sys.stdout.write("\rBuilding vocabulary: %d" % len(words))
        
        sys.stdout.flush()
        print("\rVocabulary built: %d" % len(words))

        self.words = words
        self.word_map = word_map

    def __getitem__(self, i):
        return self.words[i]

    def __len__(self):
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __contains__(self, key):
        return key in self.word_map

    def indices(self, tokens):
        return [self.word_map[token] if token in self else self.word_map['{rare}'] for token in tokens]

    def filter_for_rare_and_common(self, min_count):
        # Remove rare words and sort
        tmp = []
        tmp.append(Word('{rare}'))
        unk_hash = 0

        count_unk = 0
        for token in self.words:
            if token.count < min_count:
                count_unk += 1
                tmp[unk_hash].count += token.count
            else:
                tmp.append(token)
        
        tmp.sort(key=lambda token : token.count, reverse=True)
        
        # Update word_map
        word_map = {}
        for i,token in enumerate(tmp):
            word_map[token.word] = i
        
        self.words = tmp
        self.word_map = word_map


class TableForNegativeSamples:
    """
    非常粗暴的sample方法
    """
    def __init__(self, vocab):
        power = 0.75
        norm = sum([math.pow(t.count, power) for t in vocab])

        table_size = int(1e8)
        table = np.zeros(table_size, dtype=np.uint32)

        p = 0
        i = 0
        for j,word in enumerate(vocab):
            p += float(math.pow(word.count, power)/norm)
            while i < table_size and float(i)/table_size < p:
                table[i] = j
                i += 1
        self.table = table
    
    def sample(self, count):
        indices = np.random.randint(low=0, high=len(self.table), size=count)
        return [self.table[i] for i in indices]


def sigmoid(z):
    if z > 6:
        return 1.0
    elif z < -6:
        return 0.0
    else:
        return 1 / (1 + math.exp(-z))


def save(vocab, nn0, filename):
    f = open(filename, 'w')
    for token, vector in zip(vocab, nn0):
        word = token.word.replace(' ', '_')
        vector_str = ' '.join([str(s) for s in vector])
        f.write('%s %s\n' % (word, vector_str))
    f.close()


if __name__ == '__main__':

    input_filename = 'data/input-10000'
    
    k_negative_sampling = 5
    min_count = 3
    word_phrase_passes = 3
    word_phrase_delta = 3
    word_phrase_threshold = 1e-4

    corpus = Corpus(input_filename, word_phrase_passes, word_phrase_delta, word_phrase_threshold,'phrases-%s' % input_filename.split('/')[-1])

    vocab = Vocabulary(corpus, min_count)
    table = TableForNegativeSamples(vocab)

    window = 5
    dim = 300

    print("Training: %s-%d-%d-%d" % (input_filename, window, dim, word_phrase_passes))

    nn0 = np.random.uniform(low=-0.5/dim, high=0.5/dim, size=(len(vocab), dim))
    nn1 = np.zeros(shape=(len(vocab), dim))

    initial_alpha = 0.01

    global_word_count = 0
    alpha = initial_alpha
    word_count = 0
    last_word_count = 0

    tokens = vocab.indices(corpus)

    for token_idx,token in enumerate(tokens):
        if word_count % 10000 == 0:
            global_word_count += (word_count - last_word_count)
            last_word_count = word_count

            sys.stdout.flush()
            sys.stdout.write("\rTraining: %d of %d" % (global_word_count, len(corpus)))

        current_window = np.random.randint(low=1, high=window+1)
        context_start = max(token_idx-current_window, 0)
        context_end = min(token_idx + current_window + 1, len(tokens))
        context = tokens[context_start:token_idx]+tokens[token_idx+1:context_end]

        for context_word in context:
            neu1e = np.zeros(dim)
            classifiers = [(token, 1)] + [(target, 0) for target in table.sample(k_negative_sampling)]
            for target,label in classifiers:
                z = np.dot(nn0[context_word], nn1[target])
                p = sigmoid(z)
                g = alpha * (label - p)
                neu1e += g * nn1[target]  # Error to backpropagate to nn0
                nn1[target] += g * nn0[context_word]
            
            nn0[context_word] += neu1e
        
        word_count += 1
        
    global_word_count += (word_count - last_word_count)
    sys.stdout.flush()
    print("\rTraining finished: %d" % global_word_count)

    # Save model to file
    save(vocab, nn0, 'output-%s-%d-%d-%d' % (input_filename.split('/')[-1], window, dim, word_phrase_passes))