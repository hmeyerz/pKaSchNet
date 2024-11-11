import sys
fname = sys.argv[1]

def amber(fname): #TODO DOWNLOAD AMBER LOL, upload alternative force field, change path
    skript = f"""source /Users/jessihoernschemeyer/pKaSchNet/modified_ambers/leaprc.protein.ff14SB
    source leaprc.water.tip3p
    #loadOff UNK.lib
    mol = loadpdb {fname}
    savepdb mol "{fname}.prot"

    quit"""

    with open("ascript.py","w") as file: 
        file.writelines(skript)


amber(fname)
