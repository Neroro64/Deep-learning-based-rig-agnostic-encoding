import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as ex
import bz2
import func as F
import torch

ae_res_path = "G:/MEX/results/ae_results.pbz2"
ref_res_path = "G:/MEX/results/Reference_results.pbz2"
trans_res_path = "G:/MEX/results/transfer_results.pbz2"
ref_reduc_res_path = "G:/MEX/results/transfer_results_reduc_ref.pbz2"
ref_reduc_res_path2 = "G:/MEX/results/transfer_results_reduc_ref_2-5.pbz2"
reduc_res_path = "G:/MEX/results/transfer_results_reduc.pbz2"

def extract(data):
    df = []
    for r1 in data:
        ref_df = {}
        for k, v in r1.items():
            r1_res = pd.DataFrame()
            # r1_res["name"] = pd.Series([v["name"]])
            # r1_res["params"] = pd.Series([v["params"]])
            # r1_res["mem"] = pd.Series([v["mem"]])
            # r1_res["time"] = pd.Series([np.mean(v["elapsed_times"])])

            recon_err = [err.item() for err in v["recon_error"]]
            r1_res["recon_err"] = pd.Series(np.asarray(recon_err))
            adv_err = [err.item() for err in v["adv_error"]]
            r1_res["adv_err"] = pd.Series(np.asarray(adv_err))

            rot_err = [err.item() for err in v["rot_error"]]
            r1_res["rot_err"] = pd.Series(np.asarray(rot_err))
            sx = [sum(abs(delta[0])).item() for delta in v["delta_rot"]]
            sy = [sum(abs(delta[1])).item() for delta in v["delta_rot"]]
            sxx = pd.Series(np.asarray(sx))
            syy = pd.Series(np.asarray(sy))

            r1_res["sum_delta_rot_x"] = pd.Series([sxx])
            r1_res["sum_delta_rot_y"] = pd.Series([syy])
            pCostx = [(sum(cost[0])/float(len(cost[0])+1)) for cost in v["potCost"]]
            pCosty = [(sum(cost[1])/float(len(cost[1])+1)) for cost in v["potCost"]]
            rCostx = [(sum(cost[0])/float(len(cost[0])+1)) for cost in v["rotCost"]]
            rCosty = [(sum(cost[1])/float(len(cost[1])+1)) for cost in v["rotCost"]]

            r1_res["avg_sum_pCost_x"] = pd.Series(np.asarray(pCostx))
            r1_res["avg_sum_pCost_y"] = pd.Series(np.asarray(pCosty))
            r1_res["avg_sum_rCost_x"] = pd.Series(np.asarray(rCostx))
            r1_res["avg_sum_rCost_y"] = pd.Series(np.asarray(rCosty))

            ref_df[k] = r1_res
        df.append(ref_df)
    return df

transfer_res = F.load(trans_res_path)
dfs = extract(transfer_res)
F.save(dfs, "transfer_dfs", "../results")

reduct_res1 = F.load(ref_reduc_res_path)
reduct_res2 = F.load(ref_reduc_res_path2)
reduc_res = [reduct_res1[0]] + reduct_res2
dfs = extract(reduc_res)
F.save(dfs, "ref_reduc_dfs", "../results")
reduc_res_tr = F.load(reduc_res_path)
dfs = extract(reduc_res_tr)
F.save(dfs, "reduc_dfs", "../results")
