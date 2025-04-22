import gzip
import glob
import re
import numpy as np
import torch
import time
def find_needle(haystack, needle):
    """
    This function finds the first index in my residues ("haystack") that is equal to the first three in the pypka sequence ("needle"). 
    It is this way because it is built under the assumption that I usually have more residues than pkPDB does.
    If it does not find any matches, does the same but with mine as the needle and pkPDB as the haystack.

    note: this function does not account for chain breaks, as that should be handled by internal counters.

    m:  the tolerance of how many uninterrupted residues I need in a sequence matching to be safe. Currently, 3

    returns mstart, pstart, which are the indices which will be used to update the m_s, p_s, m_num, and p_num.
    """
    
    n, m = len(haystack), 3
    
    if n==2:
        return None, None

    # Slide a window of length m over haystack
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle[0:m]:
            return i, 0
    
    n=len(needle)
    #look for pypka seq in mine by flipping variables
    haystack,needle=needle,haystack
    
    for i in range(n - m + 1):
        if haystack[i:i + m] == needle[0:m]:
            #print(haystack[i:i + m][0],haystack[i:i + m][1])

            return 0,i
    return None, None


def first_mismatch_index(seq):
    """
    Return the first index i such that seq[i] != seq[i+1],
    or None if all consecutive pairs match (or seq has < 2 elements).
    """
    # enumerate over pairs (seq[0],seq[1]), (seq[1],seq[2]), ...
    return next(
        (i for i, (a, b) in enumerate(zip(seq, seq[1:])) if a != b),
        0
    )

def first_smaller_index(seq):
    """
    Return the first index i such that seq[i] != seq[i+1],
    or None if all consecutive pairs match (or seq has < 2 elements).
    """
    # enumerate over pairs (seq[0],seq[1]), (seq[1],seq[2]), ...
    return next(
        (i for i, (a, b) in enumerate(zip(seq, seq[1:])) if not int(a) < int(b)),
        None
    )

#this was the code used to make myids when I parsed my structures.
code = {b"H":b"0",
        b"A":b"1",
        b"L":b"2",
        b"T":b"3",
        b"G":b"4",
        b"C":b"5"}
import re
# Compile once for speed & clarity


to=time.time()
idfiles=glob.glob("/home/jrhoernschemeyer/Desktop/data_prep/nometals/ids/*.gz")
failalign=[]
badparse=[]
testset=[]
chain=False
#pdbs=["3uon","50l8"]
for c,file in enumerate(idfiles):#22]):   
    pdb=file[-7:-3]
    try:
        last_success=(0,0)
        with gzip.open(f"/home/jrhoernschemeyer/Desktop/data_prep/targets/ids/{pdb}.gz","r") as f:
            lines=f.readlines()
            f.close()
        try:
            ppypka = sorted(lines, key=lambda s: (re.match(r'(\d+)([A-Za-z])', s.decode()).group(2).lower(), int(re.match(r'(\d+)([A-Za-z])', s.decode()).group(1))))
        except:
            failalign.append(pdb)
            continue
        file=f'/home/jrhoernschemeyer/Desktop/data_prep/nometals/ids/{pdb}.gz'
        j_0=-1 
        m_0=-1
        flag=0
        counter_m,counter_p,counter2=0,0,1
        passflag=False
        
        # Pre-compile regex for efficiency.
        pattern = re.compile(r'(\d+)([A-Za-z])')
        # Build a list with (original_index, letter, number, line)
        indexed_lines = []
        for idx, line in enumerate(lines):
            decoded = line.decode().strip()  # Decode once per line.
            m = pattern.match(decoded)
            if m:
                num = int(m.group(1))
                letter = m.group(2).lower()
                indexed_lines.append((idx, letter, num, line))
            else:
                indexed_lines.append((idx, '', 0, line))
        # Sort by letter then by number.
        indexed_lines.sort(key=lambda x: (x[1], x[2]))
        # Build the mapping and extract sorted lines.
        old_to_new = {orig_idx: new_idx for new_idx, (orig_idx, letter, num, line) in enumerate(indexed_lines)}

        #mydata
        with gzip.open(file,"r") as f:
            myresis=f.readlines()
            f.close()
        myresis.sort(key=lambda b: (
        b.decode().split()[0][-1].lower(),                  # last char of first token
        int(re.match(r'\d+', b.decode().split()[0]).group())# its leading number
    ))
        
        #myresis = sorted(lines, key=lambda s: (re.match(r'(\d+)([A-Za-z])', s.decode()).group(2).lower(), int(re.match(r'(\d+)([A-Za-z])', s.decode()).group(1))))
        #myresis = sorted(lines, key=sort_key)
        #target values
        with gzip.open(f"/home/jrhoernschemeyer/Desktop/data_prep/targets/{pdb}.gz","r") as f:
            lines=f.readlines()
            f.close()
        pkas={id[:-3]:lines[old_to_new[i]].strip() for i,id in enumerate(ppypka)}#range(len(pypka))]
        #print(pkas)
        labels,ids=[],[]

        #my data
        ma=[e for e in enumerate(myresis)]
        m_nseq = [e[1].split()[0].strip(b"ABCDEFGHIJKLMNOPQRSTUVWXZY") for e in ma]
        m=[e[1].split() for e in ma]
        m_num = [e[0] for e in ma]

        #pypka
        a=[e for e in enumerate(ppypka)]
        p_nseq = [e[1].strip(b"ABCDEFGHIJKLMNOPQRSTUVWXZY \n") for e in a]
        p = [e[1].split() for e in a] #[code[e[1].split()[1]] for e in a]
        p_num = [e[0] for e in a]

        #get the sequence code for each
        m_s,p_s = [e[1] for e in m], [code[e[1]] for e in p]#,[e[0] for e in p]
        
        #get the number of residues
        lm,lp=len(m),len(p)
        jm=0
        jp=0
        ids=[]
        labels=[]
        #print("")
        #print(pdb,c)
        flag2=0
        counter2=1
        startflag=True
        #m_num and p_num are the enumerated residues 1,2,3,.....N. For each match-or missing in one and in certain siutuations, is deleted.
        while m_num and p_num:
            #print("chain",counter2)
            
            my,py=m_num[0],p_num[0] #which resi number i am in total number of resis. is like my code. enumerated resis.
            pyinfo, myinfo =ppypka[py].split(), myresis[my].split() #[b'3A', b'G'] [b'5A', b'0', b'17']
            pid,mid=np.int(pyinfo[0][:-1].strip(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")),np.int(myinfo[0][:-1].strip(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
            if pid >= j_0 and mid >= m_0:
                #start of chain or continutation
                mseq,pseq= p_s[0],m_s[0] #amino acid code
                #normal
                if lm > lp:
                    if pid == mid:
                        if mseq == pseq:
                            if int(mid) - int(last_success[0]) == int(pid) - int(last_success[1]) or passflag:
                                startflag=False
                                del m_num[0]
                                del m_s[0]
                                del p_num[0]
                                del p_s[0]
                                last_success=(mid,pid)
                                try:
                                    labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                                    ids.append(myresis[my].strip())
                                except:
                                    counter2=ppypka[py][-4] - 64
                                    ids.append(myresis[my].strip())
                                    labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                            else:
                                #print("here")
                                if startflag:
                                    startflag=False
                                    del m_num[0]
                                    del m_s[0]
                                    del p_num[0]
                                    del p_s[0]
                                    last_success=(mid,pid)
                                    try:
                                        labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                                        ids.append(myresis[my].strip())
                                    except:
                                        counter2=ppypka[py][-4] - 64
                                        ids.append(myresis[my].strip())
                                        labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                                
                                elif pid==mid:
                                    #print("very exceptional normal? match.",m_s[0],p_s[0])
                                    #startflag=False
                                    del m_num[0]
                                    del m_s[0]
                                    del p_num[0]
                                    del p_s[0]
                                    last_success=(mid,pid)
                                    try:
                                        labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                                        ids.append(myresis[my].strip())
                                    except:
                                        counter2=ppypka[py][-4] - 64
                                        ids.append(myresis[my].strip())
                                        labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                                
                                
                                else:
                                    #print("no match 1, deleting m0 and increasing conter m.")
                                    del m_num[0]
                                    del m_s[0]
                                    #counter_m += 1


                        else: #todo, mine can in theory also be missing e.g. bad parse.
                            #print("!! lm bigger than lp. and our midpid dont match. counter m go up. why do i do this?")
                            #else:
                            #print("here. no match, deleting m0 and increasing conter m.")
                            del m_num[0]
                            del m_s[0]
                            #counter_m += 1
                            #counter_m += 1

                                    

                    elif pid < mid: #prettz normal ##################################
                        #print("here2",mseq,pseq,startflag)
                        
                        if mseq==pseq:
                            #print("hereey")
                            #print(int(mid) - int(last_success[0]), int(pid) - int(last_success[1]))# int(pid) - int(j_0))
                            if int(mid) - int(last_success[0]) == int(pid) - int(last_success[1]) or passflag:
                                #print("pid < mid match",m_s[0],p_s[0])
                                passflag=False
                                #print("ms",p_num)
                                startflag=False
                            #print("normal match.",m_s[0],p_s[0])

                                last_success=(mid,pid)
                                del m_num[0]
                                del m_s[0]
                                del p_num[0]
                                del p_s[0]
                                
                                try:
                                    labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                                    ids.append(myresis[my].strip())
                                except:
                                    counter2=ppypka[py][-4] - 64
                                    ids.append(myresis[my].strip())
                                    labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                        
                            else:##########++++++++++++++++#########################
                                #mstart, pstart = find_needle(m_s, p_s)
                                m_odd, podd = first_mismatch_index(m_s),first_mismatch_index(p_s)
                                #print("new new")
                                mstart, pstart = find_needle(m_s[m_odd:], p_s[podd:])
                                if not mstart==None:
                                    m_s, p_s, m_num, p_num = m_s[m_odd+mstart:], p_s[podd + pstart:], m_num[m_odd+mstart:], p_num[podd+ pstart:]
                                    passflag=True
                                else:
                                    m_odd, podd = first_mismatch_index(m_s),first_mismatch_index(p_s)
                                    mstart, pstart = find_needle(m_s[m_odd:], p_s[podd:])
                                    if not mstart==None:
                                        m_s, p_s, m_num, p_num = m_s[m_odd+mstart:], p_s[podd + pstart:], m_num[m_odd+mstart:], p_num[podd+ pstart:]
                                        passflag=True
                                                
                                    else:
                                        #print("fail new")
                                        break  

                        elif startflag:
                            m_odd, podd = first_mismatch_index(m_s),first_mismatch_index(p_s)
                            mstart, pstart = find_needle(m_s[m_odd:], p_s[podd:])
                            if not mstart==None:
                                m_s, p_s, m_num, p_num = m_s[m_odd+mstart:], p_s[podd + pstart:], m_num[m_odd+mstart:], p_num[podd+ pstart:]
                                passflag=True   
                            else:
                                break  

                        else:
                            #print("no match n")
                            #startflag=False
                            del m_s[0]
                            del m_num[0]

                    elif pid > mid: #pypka doesnt have early residues, can be numbering also and everthing basically normal. 
                        if mseq==pseq: #
                            if int(mid) - int(last_success[0]) == int(pid) - int(last_success[1]) or passflag:
                                #print("prev sketchy exceptional match",m_s[0],p_s[0])
                                passflag=False
                                startflag=False
                                last_success=(mid,pid)
                                try:
                                    labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                                    ids.append(myresis[my].strip())
                                except:
                                    counter2=ppypka[py][-4] - 64
                                    ids.append(myresis[my].strip()) #
                                    labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                                del m_num[0]
                                del m_s[0]
                                del p_num[0]
                                del p_s[0]
                                
                            elif startflag:
                                m_odd, podd = first_mismatch_index(m_s),first_mismatch_index(p_s)
                                mstart, pstart = find_needle(m_s[m_odd:], p_s[podd:])
                                if not mstart==None:
                                    m_s, p_s, m_num, p_num = m_s[m_odd+mstart:], p_s[podd + pstart:], m_num[m_odd+mstart:], p_num[podd+ pstart:]
                                    passflag=True         
                                else: 
                                    break

                            else:
                                del m_s[0]
                                del m_num[0]

                        elif startflag:
                            m_odd, podd = first_mismatch_index(m_s),first_mismatch_index(p_s)
                            mstart, pstart = find_needle(m_s[m_odd:], p_s[podd:])
                            
                            if not mstart==None:
                                m_s, p_s, m_num, p_num = m_s[m_odd+mstart:], p_s[podd + pstart:], m_num[m_odd+mstart:], p_num[podd+ pstart:]
                                passflag=True                   
                            else:
                                break

                        else:
                            #print("no match")
                            del m_s[0]
                            del m_num[0]
                            
                elif lp == lm: 
                    if mseq == pseq: 
                        
                        if int(mid) - int(last_success[0]) == int(pid) - int(last_success[1]) or passflag:
                        
                        
                            startflag=False
                            del m_num[0]
                            del m_s[0]
                            del p_num[0]
                            del p_s[0]
                            last_success=(mid,pid)
                            try:
                                labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                                ids.append(myresis[my].strip())
                            except:
                                counter2=ppypka[py][-4] - 64
                                ids.append(myresis[my].strip())
                                labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                        else:
                            m_odd, podd = first_mismatch_index(m_s),first_mismatch_index(p_s)
                            mstart, pstart = find_needle(m_s[m_odd:], p_s[podd:])
                            if not mstart==None:
                                m_s, p_s, m_num, p_num = m_s[m_odd+mstart:], p_s[podd + pstart:], m_num[m_odd+mstart:], p_num[podd+ pstart:]
                                passflag=True
                            else:
                                m_odd, podd = first_mismatch_index(m_s),first_mismatch_index(p_s)
                                mstart, pstart = find_needle(m_s[m_odd:], p_s[podd:])
                                if not mstart==None:
                                    m_s, p_s, m_num, p_num = m_s[m_odd+mstart:], p_s[podd + pstart:], m_num[m_odd+mstart:], p_num[podd+ pstart:]
                                    passflag=True      
                                else:
                                    break  
                    else:
                        del m_num[0]
                        del m_s[0]
                        
                                            
    #############weirder situation#########################################################################################     to do try except 
                else: #lp>lm#if pypka has more resis (due to bad parse)
                    if not lm==1:
                        if pid==mid:#np.int(ppypka[i].split()[0][:-1]) == np.int(myresis[i-counter_p].split()[0][:-1]): #odd
                            if mseq==pseq:#p_s[0] == m_s[0]: #match
                                startflag=False
                                #print("match",m_s[0],p_s[0])
                                last_success=(mid,pid)
                                del m_num[0]
                                del m_s[0]
                                del p_num[0]
                                del p_s[0]
                                try:
                                    ids.append(myresis[my].strip())
                                    labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                                except:
                                    counter2=ppypka[py][-4] - 64
                                    ids.append(myresis[my].strip())
                                    labels.append(pkas[np.char.add(np.char.encode(str(pid)),chr(64+counter2)).item().encode()])
                            else:
                                #print("!!!!!!!!!!!!!!!!no match")
                                break #safe to continue? #achtung can get stuck in loop!
                            
                        elif pid >= mid:
                            #number of pypka resis is still higher than myresis. prob means parsing error mixed with added/resis not calculated for.
                            del m_num[0]
                            del m_s[0] 
                            
                        else: 
                            del p_num[0]
                            del p_s[0]
                            #badparse.append(pdb)
                        
                    else:
                        #dont do anything cuy cant align one
                        break
                
            elif mid < m_0 and pid > j_0:#or m_0 <  
                #my had chain break, py not
                #print("mine has chain beak")
                idx = first_smaller_index(p_nseq[py:])
                if not idx==None:
                    p_s=p_s[idx+1:]
                    p_num=p_num[idx+1:]
                else:
                    break
                    

            elif pid < j_0 and mid > j_0: #pypka has chain breaked and mine, not.
                idx = first_smaller_index(m_nseq[my:])
                if not idx==None:
                    m_s=m_s[idx+1:]
                    m_num=m_num[idx+1:]
                    counter_m+=idx
                else:
                    break
                
            elif pid <= j_0 and mid <= m_0: #chainbreak
                counter2+=1
                #print("chain, m_0 j_0",m_0,j_0)
                chain=True
                m_0,j_0 = -1,-1
                last_success=(0,0)
                startflag=True
        
            if not chain: #if this isnt a pass with a chain
                j_0, m_0 = pid,mid
                chain=False

        

            #print(pdb, j_0,m_0)
    
        if len(ids)>lp:
            print("lebens gefahr")
            continue
        
        if ids: 
            if len(ids) < .5*lp:
                testset.append(pdb)
            else:
                with gzip.open(f"/home/jrhoernschemeyer/Desktop/data_prep/nometals/aligned/ids/{pdb}.gz", "wb") as f:
                    for id in ids:
                        f.write(np.char.add(id,b"\n"))
                
                torch.save(torch.tensor([np.float32(l) for l in labels]),f"/home/jrhoernschemeyer/Desktop/data_prep/nometals/aligned/targets/{pdb}")

        
        else:
            failalign.append(pdb)
        #print(counter_m, counter_p, "l1",len(m_s),lm,"l2",len(p_s),lp)
        
        
    except:
        failalign.append(pdb)

with gzip.open("/home/jrhoernschemeyer/Desktop/data_prep/nometals/failedalign.gz","wb") as f:
    for path in failalign:
        f.write(f"{path}\n".encode())
with gzip.open("/home/jrhoernschemeyer/Desktop/data_prep/nometals/failalign_testset.gz","wb") as f:
    for path in testset:
        f.write(f"{path}\n".encode())
                    #f.close()
    
#print(len(labels),len(ids))
print("done aligning mins", (to - time.time())/60)