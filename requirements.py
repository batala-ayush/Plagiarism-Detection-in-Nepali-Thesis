class tokenize:
    def __init__(self,text):
        self.text = text
    def word_tokenize(self):
        punctuations = ['।', ',', ';', '?', '!', '—', '-', '.']
        for punctuation in punctuations:
            self.text = self.text.replace(punctuation, ' ')


        self.text = self.text.split()
        return self.text

t = tokenize("Hey, Copines; is। a good? song; I! like— that- song. and")
print(t.word_tokenize())