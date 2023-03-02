package edu.mcgill.cstk.experiments.repair

import ai.hypergraph.kaliningraph.levenshtein
import ai.hypergraph.kaliningraph.parsing.*
import edu.mcgill.cstk.utils.*

val pythonStatementCFG: CFG = """
S -> w | w ( S ) | ( ) | S = S | S . S | S S | ( S ) | [ S ] | { S } | :
S -> S , S | S ; S | S : S
S -> S + S | S - S | S * S | S / S | S % S
S -> S < S | S > S | S <= S | S >= S | S == S | S != S
""".trimIndent().parseCFG()
  .apply { blocked.addAll(terminals.filter { !it.isBracket() })  }

val testStatements = """
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

val testInvalidStatements = """
  expectedReducedC = [(i, sum(map(lambda(x, y): y, xs))) for(i, xs) in expectedGroupedC]
  s = '0 ' + ' '.join(map(lambda(i, v): '%d:%d' %(i + 1, v), enumerate([i for i in range(20)])))
  d = sum(map(lambda(a, b): abs(a - b), zip(stringshist[i], Sshist)))
  sql = 'insert into ' % s ' (%s) values (%s)' %(table, ','.join(['' % s '' % col for col in cols]), ','.join(['?' for i in range(len(cols))]))
  result = array(map(lambda(x, y): [x, y[0], y[1]], self.run_ids_dict.iteritems()))
  dt_df = sqlContext.createDataFrame(data.rdd.map(lambda(k, v): list(itertools.chain(*[[k], [float(x) for x in list(v)]]))), ['0', '1', '2', '3'])
  d = sum([abs(rgb1[i] - rgb0[i]) for i in 0, 1, 2])
  data = map(lambda(l, f): (int(l), int(f[0])), data)
  a, b = [(0 if(x[0] >= 0.9) else x[0], x[1], x[2] * 0.3) for x in a, b]
  blockobjs = blocks.map(lambda(n, c): (n, Block.of_string(c.strip(), 0)[0]))
  elif i == 2: values.append(stringify(redact_urls(max(words_ws[- j - 1].iteritems(), key = itemgetter(1))[0])))
  area = sum([float(trapezoidal_area(X[k], Y[k] X[k_minus_one], Y[k_minus_one])) for k, k_minus_one in indices_to_consider])
  wcv = closest.map(lambda(k, v): np.sum((centroids[k] - v[0]) ** 2)).reduce(lambda x, y: x + y)
  s = ''.join([_struct.pack("IIII", * cpuid(EXTENDED_OFFSET | k)) for k in 0x2, 0x3, 0x4])
  'links': [{'rel': 'subsection', 'type': 'application/atom+xml', 'href': reverse('by_tag_feed', kwargs = dict(tag = tag.name)}]}
  __makeGetUrl = lambda(x): '/' + '/'.join([x['uid'], datetime.strptime(x['date'], "%Y-%m-%dT%H:%M:%S.%f").strftime("%Y-%m-%d")])
  diff = [(bin[0], bin[1] - hist2[bin[0]]) for bin in hist1.items() if bin[0] in hist2] git
  edir, edn, eup = [dfield.reshape((np.max(z) + 1, np.max(y) + 1, np.max(x) + 1)) for dfield in edir, edn, eup]
  r = tuple([int(a[i] *(1 - v) + b[i] * v) for i in 0, 1, 2])
""".trimIndent()

/*
./gradlew pythonStatementRepair
 */

fun main() {
  testStatements.lines().filter { it.isNotBlank() }.forEach {
    val original = it.tokenizeAsPython().joinToString(" ")
    val prompt = original.constructPromptByDeletingRandomBrackets(1)
    println("Original:  $original\nCorrupted: ${prettyDiffNoFrills(original, prompt)}")
//  testInvalidStatements.lines().filter { it.isNotBlank() }.forEach {
//    val prompt = it.tokenizeAsPython().joinToString(" ") // No need to corrupt since these are already broken
    MAX_SAMPLE = 3
    repair(
      prompt = prompt,
      cfg = pythonStatementCFG,
      coarsen = String::coarsenAsPython,
      uncoarsen = String::uncoarsenAsPython,
      synthesizer = { a -> a.solve(this) },
      diagnostic = { println("Δ=${levenshtein(prompt, it) - 1} repair: ${prettyDiffNoFrills(prompt, it)}") },
      filter = { isValidPython() }
    ).also { println("Original string was ${if(original in it) "" else "NOT" }contained in repairs!") }

    println("\n")
  }
}
