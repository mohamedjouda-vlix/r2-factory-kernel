import json
from multiprocessing import Pool
from experiments import run_conwip
if __name__=="__main__":
    r=json.load(open("results.json"))
    jobs=[(w,s,k,2.0e4,1.0e3) for w in range(1,13) for s in (0.0,0.25,1.0) for k in range(10)]
    with Pool(2) as p: r["conwip"]=p.map(run_conwip,jobs)
    json.dump(r,open("results.json","w"))
