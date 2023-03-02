package edu.mcgill.cstk.experiments.repair

import com.beust.klaxon.Klaxon
import java.io.File

/*
./gradlew extractRepairSamples
 */
fun main() {
  var i = 0
  val json = File("bifi/data/orig_good_code/orig.good.json")
    .readLines().takeWhile { if (it == "  },") i++ < 20000 else true }
    .joinToString("\n") + "\n}}"

  val goodCode = Klaxon().parse<Map<String, Map<String, Any>>>(json)

  goodCode!!.values.take(1000)
    .map { cs -> cs["code_string"].toString() }.forEach {
      // Check whether it matches pattern "x = [ ... ]
      it.lines().filter { " = " in it }
        .filter { it.isANontrivialStatementWithBalancedBrackets(2, statementCriteria = { true }) }
        .forEach { line -> println(line.trim()) }
    }
}

/**
pattern = re.compile(r'\d{1,3}(?=(\d{3})+(?!\d))')
calibrated = int(self.calib[0] +(level *(self.calib[1] - self.calib[0])) / 100.0)
i[...] = header['HIERARCH ESO DRS CAL TH COEFF LL{0}'.format(str(int(i)))]
url_re = re.compile(r'(?im)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\([^\s()<>]+\))+(?:\([^\s()<>]+\)|[^\s`!()\[\]{};:\'".,<>?]))')
placeholder_re = re.compile(r'{([a-zA-Z]+\d*=?[^\s\[\]\{\}=]*)}')
placeholder_content_re = re.compile(r'^(?P<placeholder_name>[a-zA-Z]+)(\d*)(=[^\s\[\]\{\}=]*)?$')
bbcodde_standard_re = r'^\[(?P<start_name>[^\s=\[\]]*)(=\{[a-zA-Z]+\d*=?[^\s\[\]\{\}=]*\})?\]\{[a-zA-Z]+\d*=?[^\s\[\]\{\}=]*\}(\[/(?P<end_name>[^\s=\[\]]*)\])?$'
bbcodde_standalone_re = r'^\[(?P<start_name>[^\s=\[\]]*)(=\{[a-zA-Z]+\d*=?[^\s\[\]\{\}=]*\})?\]\{?[a-zA-Z]*\d*=?[^\s\[\]\{\}=]*\}?$'
__version__ = ('.'.join(map(str, VERSION[: 3])) + '.'.join(VERSION[3: ]))
ids = list(set([fn.split('.')[0] for fn in filenames if fn != 'lock']))
origin = np.array(list(reversed(itkimage.GetOrigin())))
spacing = np.array(list(reversed(itkimage.GetSpacing())))
 */