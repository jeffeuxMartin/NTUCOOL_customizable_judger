# NTUCOOL_customizable_judger

Customizable quiz judger of NTU COOL.

## Usage

```bash
python3 src/general_judger.py \
        SOURCE_CSV_FILE \
        OUTPUT_GRADE_CSV \
        [STARNDARD_SOLUTION_JSON]
```

Note that if `STARNDARD_SOLUTION_JSON` is not given, the default file name will be `standard.json`.

To check if the judger works normally, the grading of each possible response to each problem will be printed out. A better practice is 
```bash
python3 src/general_judger.py \
        SOURCE_CSV_FILE \
        OUTPUT_GRADE_CSV \
        [STARNDARD_SOLUTION_JSON]
		> OUTPUT_LOG
```
Then, you can see how many points are given to each possible response in each problem.

If any special adjustment is necessary, you can change the part of 
```python
# region -------------- ADJUST -------------------- #
# ...
# endregion
```
to fit your requirement.

Also note that one and *only one* *standard answer* is referenced in this version. That is, some person needs to give an entirely correct submission. A version without necessity of the standard answer may be released later.
