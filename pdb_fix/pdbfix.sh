#for file in /home/pbuser/Desktop/Jessi/data_prep/PDB; do
for file in /Users/jessihoernschemeyer/pKaSchNet/PDB/*.ent; do
    python3 /Users/jessihoernschemeyer/pKaSchNet/fix_pdb.py "$file" #pdbfixer
    rm $file
done
    #reduce -BUILD -NUC -NOFLIP $file-fixed.pdb > $file-fixed.pdb
    #pdb4amber -i $f-Hfixed -o $pdb 



