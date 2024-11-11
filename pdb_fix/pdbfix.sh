for file in /Users/jessihoernschemeyer/pKaSchNet/PDB/*.ent; do
#for file in /home/pbuser/Desktop/Jessi/data_prep/PDB/*.ent; do
    #python3 /home/pbuser/Desktop/Jessi/data_prep/fix_pdb.py "$file" #pdbfixer
    python3 /Users/jessihoernschemeyer/pKaSchNet/fix_pdb.py "$file" #pdbfixer
    reduce -BUILD -NUC -NOFLIP -Quiet $file-fix.pdb > $file
    pdb4amber -i $file -o $file.pdb
    python3 /Users/jessihoernschemeyer/pKaSchNet/generate_amber_script.py "$file.pdb" 
    #tleap -s -f /Users/jessihoernschemeyer/pKaSchNet/ascript.py > leap.log 2>&1 #normal
    #rm $file
done




