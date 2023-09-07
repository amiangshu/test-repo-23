def compute_average_diagonal(matrix, dimension):
    sum =0.0
    for i in range(0, dimension):
        sum += matrix[i][i]
    average =sum/dimension #comput average
    return  average

def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))


def replace_bigram(texts):
    bigram = gensim.models.Phrases(texts, min_count=20, threshold=10) #find bigrams that appear at least 10 times.
    mod = [bigram[sent] for sent in texts]
    return mod


def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def remove_stopwords(tokens):
    stopped_tokens = [i for i in tokens if not i in stemmed_stopwords]
    return stopped_tokens #return token list


def remove_domainterms(tokens):
    newtokens = [i for i in tokens if not i in stemmed_domain_terms]  # remove domain terms
    return newtokens


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [w.lower() for w in tokens if w.isalpha() == True or '_' in w]  # lower case, remove number, punctuation
    return tokens


url_regex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def remove_url(s):
    return url_regex.sub(" ", s)


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    cleantext = html.unescape(cleantext) #I am removing HTML markups
    return cleantext


def cleanup_text(text):
    # comments=text
    text = str(text).encode('ascii', 'ignore').decode('ascii')
    text = cleanhtml(text)  # clean html
    text = remove_url(text)  # remove url
    return text


def preprocess_text(text):
    # comments=text
    tokens = tokenize(text)
    tokens = stem_tokens(tokens)
    tokens = remove_stopwords(tokens)
    tokens = remove_domainterms(tokens)
    return tokens


def get_random_number():
    return random.randint(0, 50000)


class LDADocument:
    def __init__(self, id, posttype, body):
        self.id = id
        self.posttype = posttype
        self.body = body

