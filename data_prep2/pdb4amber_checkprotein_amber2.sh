#TODO CHECK FOR SALTS AND DELETE!!!!!!!

#TODO EDIT substring (and add 3 for .ent), change paths

for file in /Users/jessihoernschemeyer/pKaSchNet/PDB/*.ent; do
     python3 /home/pbuser/Desktop/Jessi/data_prep/checkpdb.py "$file"
     /home/pbuser/Desktop/Jessi/data_prep/
     if [ -e "$file" ]; then
          substring=${file:40:4} 
          file2="/home/pbuser/Desktop/Jessi/data_prep/PDB/pdb4amber/$substring"
          pdb4amber -i "$file" -o "$file2" --reduce --most-populous -a  [-options] 2> pdb4amber.log
          python3 /home/pbuser/Desktop/Jessi/data_prep/generate_amber_script.py "$file2" 
          tleap -s -f /home/pbuser/Desktop/Jessi/data_prep/ascript.py > leap.log 2>&1 

          #rm $file #TODO #LAST THING
     else
          continue
     fi
done


