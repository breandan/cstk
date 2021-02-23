# gym-fs

A fast, RL environment for the filesystem.

Stores BPE-compressed files in memory.

Gives an agent the ability to selectively read files.

Interface:

* `Path.read(start, end)` - Returns file chunk at offset.
* `Path.grep(query)` - Returns offsets matching query.

# Usage

Manual grep: `./gradlew Loader --args='--query=<QUERY>'`