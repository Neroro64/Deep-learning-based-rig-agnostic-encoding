import os
def exp1():
    import transfer_models
    transfer_models.main()
R1_ZIN = exp1

def exp2():
    import transfer_models_raw
    transfer_models_raw.main()
R1_ZIN_RAW = exp2

def exp3():
    import transfer_models_reduc
    transfer_models_reduc.main()
R1_ZIN_REDUC = exp3

def exp4():
    import transfer_models_reduc_raw
    transfer_models_reduc_raw.main()
R1_ZIN_REDUC_RAW = exp4

def exp5():
    import transfer_models_cat
    transfer_models_cat.main()
R1_ZCAT = exp5

def exp6():
    import transfer_models_cat_raw
    transfer_models_cat_raw.main()
R1_ZCAT_RAW = exp6

def exp7():
    import transfer_models_cat_reduc
    transfer_models_cat_reduc.main()
R1_ZCAT_REDUC = exp7

def exp8():
    import transfer_models_cat_reduc_raw
    transfer_models_cat_reduc_raw.main()
R1_ZCAT_REDUC_RAW = exp8

def exp9():
    import train_ref_models
    train_ref_models.main()
REF_MODEL = exp9

def run(ex, id):
    try:
        ex()
        print("Done " + str(id))
    except:
        print("Failed " + str(id))


# run(R1_ZIN, 1)
# run(R1_ZIN_RAW, 2)
# run(R1_ZIN_REDUC, 3)
# run(R1_ZIN_REDUC_RAW, 4)

# run(REF_MODEL, 4)
run(R1_ZCAT, 5)
run(R1_ZCAT_RAW, 6)
# run(R1_ZCAT_REDUC, 7)
# run(R1_ZCAT_REDUC_RAW, 8)
