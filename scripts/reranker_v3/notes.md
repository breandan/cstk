Launch Modal training:

```bash
modal run train_modal_reranker.py --steps 200000
```

Fetch WebGPU artifacts:

```bash
modal volume get ranker-artifacts reranker.safetensors . --force && modal volume get ranker-artifacts reranker.js . -- force
```

Expected benchmark results:

```
====== BENCHMARK RESULTS ======
Model docs/call   : 64
Total docs scored : 1000  (chunked in 16 calls)
Iterations        : 10
Avg latency       : 1201.12 ms (1 query vs 1000 docs)

====== QUERY ======
"x = y + z E?BOT78w4cQ"

====== TOP 10 DOCUMENTS & SCORES ======
[Score:  -1.1961]  Doc 790: ewvi)KKkMIImiyR1YhIoXVeQx!1IwFZX)mxmARuH2;7G-pGMDxSD.)RNKlvzi8Ut43rBng03BrQln90
[Score:  -1.2083]  Doc 104: NKCDMWJN9ln8k.ruAS1f9xYfi4YM6by4H7L5)0Qi(XQ5mZrvk9rroN9BDW?a.AXFbWY1!xKrN6EGj
[Score:  -1.2217]  Doc 776: bZhSDRZ)q?9t7MszpAVY-1)KM JHMd)Zwg-QITJk!VvF5brBNVFDYrZZ6aB6R.B2).EjdOGPKjp7
[Score:  -1.2306]  Doc 317: fogyR?d2725v642O2te4dD.YtYY ;8iKskEaAqKtAPje)SQLpt0oJuYOWm)tBGQ)JXa-?IdrCOR0dm
[Score:  -1.2341]  Doc 662: :UKbsUmif gXL(z1rwjAMVvNfYSWYRvY)AbS!Aao(l4(qiwcAaLWSUQISjG vr4-Ca;bs9(YTE9bYc
[Score:  -1.2352]  Doc 717: R 2Wr6Jn,T80hC).HnTRUOYW!azJ7WSIp?l;xoXsGN)UittMG6a6X0JaYrhssLjjA78POHgTZ;bVbtw
[Score:  -1.2361]  Doc 750: BD,7;9:pQgH:pEiMa?8JpOHpdqukXs5oEs!y!zcC:APp2d!:COge9)CtNIPvOTOGmvsjQkyldp:6yPh
[Score:  -1.2386]  Doc 329: D2aaZ)9QQJQ2cqgG(2KAywiHOPP7W.PqC:Mjl1dgxhUgvguHnjBHfumF!k9Y)Ec6Ke0u)OBFL
[Score:  -1.2430]  Doc 961: R4?Wyq(22j4daaztS5g2vs281b6dxYagT,9.,1D7:y)s5a(l jsG8G2w1ZTDM-epRVMcs00gklGLy8Z
[Score:  -1.2431]  Doc 657: 2Y)Xucr6elu)?Gz9HGaoZsFpI,c2u)7WGUCjPvnsWb,Eo,ItmAb5oyW0PxHBJNL NyPH;d,SQV!sZf
```