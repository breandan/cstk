package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import edu.mcgill.cstk.utils.*
import ai.hypergraph.kaliningraph.parsing.repair
import org.intellij.lang.annotations.Language

/*
./gradlew pythonStatementRepair
*/

fun main() {
  MAX_SAMPLE = 3
  // Synthetic error correction
//    validPythonStatements.lines().filter { it.isNotBlank() }.forEach {
//    val original = it.tokenizeAsPython().joinToString(" ")
//    val prompt = original.constructPromptByDeletingRandomBrackets(1)
//    println("Original:  $original\nCorrupted: ${prettyDiffNoFrills(original, prompt)}")
//    // Organic repairs
//    repairPythonStatement(prompt)
//      .also { println("Original string was ${if(original in it) "" else "NOT " }contained in repairs!") }
//    println("\n")
//  }

  // Organic error correction
  invalidPythonStatements.lines().filter { it.isNotBlank() }.forEach {
    val prompt = it.tokenizeAsPython().joinToString(" ") // No need to corrupt since these are already broken
    repairPythonStatement(prompt)
    println("\n")
  }
}

fun repairPythonStatement(prompt: String): List<Σᐩ> = repair(
  prompt = prompt,
  cfg = pythonStatementCFG,
  coarsen = String::coarsenAsPython,
  uncoarsen = String::uncoarsenAsPython,
  synthesizer = { a -> a.solve(this) },
  diagnostic = { println("Δ=${levenshtein(prompt, it) - 1} repair: ${prettyDiffNoFrills(prompt, it)}") },
  filter = { isValidPython() }
)

val pythonStatementCFG: CFG = """
S -> w | w ( S ) | ( ) | S = S | S . S | S S | ( S ) | [ S ] | { S } | : | * S
S -> S , S | S ; S | S : S
S -> S + S | S - S | S * S | S / S | S % S | S ** S
S -> S < S | S > S | S <= S | S >= S | S == S | S != S
""".trimIndent().parseCFG()
  .apply { blocked.addAll(terminals.filter { !it.isBracket() })  }

@Language("py")
val testValidStatements = """
  values = sorted(set([(n - i - 1) * a + i * b for i in range(n)]))
  calibrated = int(self.calib[0] +(level *(self.calib[1] - self.calib[0])) / 100.0)
  mask = np.array([(o.can_init(obs) and o.pi(obs) == a) for o in self.options])
  N = sum([i for(_, i) in list(corpus.items())])
  p = [Philosopher(i, c[i], c[(i + 1) % n], butler) for i in range(n)]
  R = np.array([[cos(angle), - sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]])
  l = sum([len(str(s)) for s in self.segments])
  seen_authors = set(filter(proper_name, (t[0] for t in seen)))
  pas = str((i * temps) /(len(x[: , 0])))
  ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), os.pardir))
  tagged_data = set([(row[0], correct_tag) for row in reader])
  pDiffL = sum([evidence2([x], a, b) for x in s], axis = 0)
  mask = ~(2 **(32 - int(network[1])) - 1)
  val_deci = sng *(val_list[0] +((val_list[1] +(val_list[2] / 60.0)) / 60.0))
  X_b = np.hstack((np.ones((X.shape[0], 1)), X))
  __all__ = list(module for _, module, _ in pkgutil.iter_modules([os.path.split(__file__)[0]]))
  tmpl_map = dict([(int(k), t) for k, t in tmpls])
  return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X) / 2.0], axis = 1)
  farm[mill] = dict(farm.get(mill, {}), **{day: farm.get(mill, {}).get(day, 0) + int(prod)})
""".trimIndent()

@Language("py")
val invalidPythonStatements = """
numValues = sum([len(i.cache.keys()) for i in _memoizedFunctions]),
expectedGroupedC = [(i, [(i, i * 3 + j) for j in range(3)]) for i in range(5)]
res2 = array(map(lambda x: int(x[1]), tmp))
val = np.array([(1698 - 10.79 * xi) *(np.abs(np.cos(- 0.11 +(xi + 1) / 6)) - 1) + 484 for xi in x])
bottom_ninetyfive_percent = sorted(signal)[: int(np.floor(len(signal) * 0.95))]
data = concatenate((data[0: indx[0] - 1, : ], data[indx[0] + 1: data.shape[0] - 1, : ]))
indx = (s[argmin(s[: , 1]), 0] if d_max > 6 * amin(s[: , 1]) else 1)
sbar_dudy = std(concatenate((data[indx: m - 1, 1], data[indx: m - 1, 2]))) / sqrt(ar_dudyI0.eff_N[0] + ar_dudyIn.eff_N[0])
total = sum([int(input().split()[target_idx]) for _ in range(N)])
val = np.array([(abs(b *((- xi) / a + 1))) *(- 1) *(np.abs(np.cos(c * xi)) - 1) ** 2 + d for xi in x])
sstot = sum([(Y[i] - np.mean(Y)) ** 2 for i in xrange(len(Y))])
sserr = sum([(Y[i] - fun(X[i], * args)) ** 2 for i in xrange(len(Y))])
delivery = sender.send(Message(body = unicode(self.cookies[randint(0, self.upper)])))
index = int(x[8: 10]) * 12 + int(math.floor(int(x[10: 12]) / 5))
self.data = dict(map(lambda k: (k, CacheElement(dicttocache[k], self, k)), dicttocache))
self.curtar = np.array([img[int(lbl / 2)]])
i = (np.array([np.nonzero(r <= C)]))[0][0][0]
diff = round(((float(row[11]) - float(default_time[key])) / float(default_time[key])) * 100, 2)
s2 = sum([sum([int(unit) for unit in str(int(digit) * 2)]) for pos, digit in enumerate(number) if not int(pos) % 2])
weight = scipy.sum(atlas_image[objects[rid - 1]][label_image[objects[rid - 1]] == rid])
hsize = int((float(img.size[1]) * float(wpercent)))
station_ids = sorted((int(child[0].text) for child in root[1: ]))
ppts1 = array([[p.x(), p.y()] for p in pts1])
ppts2 = array([[p.x(), p.y()] for p in pts2])
ppts1 = array([[p.x(), p.y()] for p in pts1 + pts1[0: 1]])
ppts2 = array([[p.x(), p.y()] for p in pts2 + pts2[0: 1]])
ppts1 = array([[p.x(), p.y()] for p in shape1 + shape1[0: 1]])
ppts2 = array([[p.x(), p.y()] for p in shape2 + shape2[0: 1]])
sa1 = array([[p.x(), p.y()] for p in result[2] + result[2][0: 1]])
sa2 = array([[p.x(), p.y()] for p in result[3] + result[3][0: 1]])
ss1 = array([[p.x(), p.y()] for p in segs1 + segs1[0: 1]])
ss2 = array([[p.x(), p.y()] for p in segs2 + segs2[0: 1]])
ss1 = array([[p.x(), p.y()] for p in segs1 + segs1[0: 1]])
ss1 = array([[p.x(), p.y()] for p in segs1 + segs1[0: 1]])
ss1 = array([[p.x(), p.y()] for p in segs1 + segs1[0: 1]])
Float = min(max(float(self.inputs[0].sv_get()[0][0]), self.minim), self.maxim)
(var, value) = (string[: i], string[(i + len(op)): ])
targets = np.array([float(row[- 1]) for row in reader])
bVectors = np.array([(np.cos(t), np.sin(t)) for t in starters])
available_moves = [(Coordinate(0, 0), Board.from_grid([[]]))]
distance = np.linalg.norm(list(np.asarray(points[i]) - np.asarray(points[j])));
signtx.vin[0].scriptSig = CScript([- 1, OP_CHECKSEQUENCEVERIFY, OP_DROP] + list(CScript(signtx.vin[0].scriptSig)))
z = [List([b[0]] +[y.v[i] for y in b[1: ]]).Eval(env, block) for i in range(n)]
RV = [[R *(finish[y] - start[y]) /(L + 0.0000001), R *(start[x] - finish[x]) /(L + 0.0000001)]]
SA = [[(start[x] - RV[0][x]), (start[y] - RV[0][y]), 0]]
SB = [[(start[x] + RV[0][x]), (start[y] + RV[0][y]), 0]]
FA = [[(finish[x] - RV[0][x]), (finish[y] - RV[0][y]), 0]]
FB = [[(finish[x] + RV[0][x]), (finish[y] + RV[0][y]), 0]]
print(len([r for r in treestore.get_names(tree_uri = args.uri, format = None)]))
activeBitsFeature = np.array(list(objectSDRs[objectNames[i]][j][0][1]))
activeBitsLocation = np.array(list(objectSDRs[objectNames[i]][j][0][0]))
total_orders = sum([float(row[4][1: ]) for row in data])
loop.run_until_complete(gather(*[check_img(filenames = filenames, post = post) for post in posts], return_exceptions = True))
name_change = not u.data[- 1] or ALIAS_TIME /((float(u.data[- 1]) / ALIAS_DAYS) ** 2) < time_since_crawl
if i == 1: values.append(stringify(redact_urls(max(cleaned_names_ws[- j - 1].iteritems(), key = itemgetter(1))[0])))
distanceArray = np.array([np.sum((X[m, : ] - X[k, : ]) *(X[m, : ] - X[k, : ])) for m in groupIndexArray[kmeans[k]][0]])
distanceArray = np.array([np.sum((X[m, : ] - X[k, : ]) *(X[m, : ] - X[k, : ])) for m in groupIndexArray[kmeans[k]][0]])
masses3_sqrt1 = np.array(sum([[1 / m, 1 / m, 1 / m] for m in np.sqrt(masses)], []))
buckets = dict(map(lambda n: (n, []), range(num_files)))
log_bleu_prec = sum([math.log(float(x) /(y)) for x, y in zip(stats[2: : 2], stats[3: : 2])]) / 4.
PairwiseSlopes = np.append(PairwiseSlopes, ((TheData[r] - ii) /(r - i)))
nData = len(list(TheData[np.where(TheData > TheMDI)]))
suffix = [elem for elem, _ in state[state.index((file, time)): ] if(not(elem in modified or elem in removed))]
row = round((y -(col_f * self.spacing[1] / 2) - self.orig[1]) / self.spacing[1])
lhsDesign = ot.LHSExperiment(ot.ComposedDistribution([ot.Uniform(0.0, 1.0)] * dimension), size)
self._macro_to_pc[self._macro_to_pc.index(p)] = ((p[0], int(p[0].value), p[2], p[3]))
rotated = numpy.dot(transform, numpy.array([vector[0], vector[1], vector[2], 1.0]))[: 3]
line = new_format.format(*(re.split(SEP, line[1: ])))
points = [(long(x[i]), long(self.height - y[i])) for i in range(len(x))]
res = np.insert(res, 0, [1 for j in xrange(X.shape[1])], axis = 0)
prob = clf.predict_proba(np.arrat([features[feature]]))
self.z_etoile_barre = array([g(self.points_sigma[i]) for i in range(0, 2 * self.d)])
g3 = Vector({E[1]: g3_num.dict[E[1]] / g3_den, E[3]: g3_num.dict[E[3]] / g3_den})
y = cy +((cos_table[(angle + 90) % 360] * radius) >> COSFP)
ret = process_f([(self.env, config_override[key])])
points = [Point((p[0] * App.gsx + ox) * sx, (p[1] * App.gsy + oy) * sy) for p in ps]
data_ = [(n.mean(topom_dict[i]), n.std(topm_dict[i])) for i in vertex_measures]
data_ = [(n.mean(topom_dict[i][j]), n.std(topm_dict[i][j])) for i in sector_vertex_measures for j in range(3)]
info = dict([(k.upper(), v) for k, v in info_.items()])
checked_param.eval(feed_dict = {param: np.ones([1])})
return 1. / thish ** self._dim * numpy.sum(numpy.tile(self._w, (x.shape[0], 1)) * thiskernel / self._lambda[: , 0] ** self._dim, axis = 1)
T = array([[0], [0], [k * np.sum([i for i in inputs])]])
""".trimIndent()