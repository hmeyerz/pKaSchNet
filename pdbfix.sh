for file in /Users/jessihoernschemeyer/pKaSchNet/PDB/*.ent; do
#for file in /home/pbuser/Desktop/Jessi/data_prep/PDB/.ent; do

    pdb=${file: -8:4}
    echo $pdb
    #python3 /home/pbuser/Desktop/Jessi/data_prep/fix_pdb.py "$file" #pdbfixer
    python3 /Users/jessihoernschemeyer/pKaSchNet/fix_pdb.py "$pdb" #pdbfixer #xmap, ymap1
    echo "fixedd"

    reduce -BUILD -NUC -NOFLIP -Quiet $file-fix > /Users/jessihoernschemeyer/pKaSchNet/PDB/$pdb-red
    echo "reduced"

    pdb2pqr --ff AMBER --with-ph 7.0 /Users/jessihoernschemeyer/pKaSchNet/PDB/$pdb-red /Users/jessihoernschemeyer/pKaSchNet/$pdb.pqr 

done




