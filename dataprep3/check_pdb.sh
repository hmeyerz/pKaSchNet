for file in /home/pbuser/Desktop/Jessi/data_prep/PDB/*.ent; do
    python3 /home/pbuser/Desktop/Jessi/data_prep/checkpdb.py "$file"
done
