for file in /home/pbuser/Desktop/Jessi/data_prep/PDB/*.ent; do

    pdb=${file: -8:4}
    echo $pdb
    python3 /home/pbuser/Desktop/Jessi/data_prep/fix_pdb.py "$pdb" #pdbfixer
    reduce -BUILD -NUC -HIS -Quiet /home/pbuser/Desktop/Jessi/data_prep/PDB/$pdb-fixed.pdb > /home/pbuser/Desktop/Jessi/data_prep/PDB/$pdb-red.pdb
    #pdb2pqr --ff AMBER --keep-chain --assign-only --log-level WARNING --quiet /home/pbuser/Desktop/Jessi/data_prep/PDB/$pdb-red.pdb /home/pbuser/Desktop/Jessi/data_prep/PDB/$pdb.pqr 

done




