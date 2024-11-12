for file in /Users/jessihoernschemeyer/pKaSchNet/PDB/*9l.ent; do
#for file in /home/pbuser/Desktop/Jessi/data_prep/PDB/*9l.ent; do
    echo $file
    #python3 /home/pbuser/Desktop/Jessi/data_prep/fix_pdb.py "$file" #pdbfixer
    #python3 /Users/jessihoernschemeyer/pKaSchNet/fix_pdb.py "$file" #pdbfixer
    #echo "fixed"
    #reduce -BUILD -NUC -NOFLIP -Quiet $file-fix.pdb > $file.1.pdb
    #echo "reduced"
    #pdb4amber -i $file.1.pdb -o $file.pdb
    #echo "pdb4ambered"
    python3 /Users/jessihoernschemeyer/pKaSchNet/generate_amber_script.py "$file.pdb" 
    tleap -s -f /Users/jessihoernschemeyer/pKaSchNet/ascript.py > leap.log 2>&1 #normal
    #rm $file
done




