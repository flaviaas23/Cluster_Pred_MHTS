
#codigo exemplo de orientacao à objeto
import os

class doce:
    def __init__(self):
        self.sabor = ''
    
    def esta_bom(self):
        return True

class biscoito(doce):
    count_biscoitos=0

    @staticmethod
    def total_biscoitos():
        return biscoito.count_biscoitos

    def __init__(self):
        super(biscoito, self).__init__()
        self.formato=''
        self.__pronto__ = False
        biscoito.count_biscoitos += 1

    def assar_biscoito(self):
        self.__pronto__=True
    
    def esta_pronto(self):
        if self.__pronto__:
            return True
        else: 
            return False
    
    def esta_bom(self):
        return False

if __name__=='__main__':
    b1=biscoito()
    b2=biscoito()

    print (b1.esta_bom())
    print(biscoito.count_biscoitos)
    print(biscoito.total_biscoitos())

    #b1.sabor='chocolate'
    b1.formato='bola'

    #b1.assar_biscoito()

    assou = b1.esta_pronto()

    print ("assou: ", assou)
    print ("sabor: ", b1.sabor)
    print (os.getcwd())





