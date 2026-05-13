import pandas as pd
import re

raw_train = """
t = 1	0.1701	t = 26	0.2398	t = 51	0.3087	t = 76	0.3701
t = 2	0.1023	t = 27	0.0508	t = 52	0.0159	t = 77	0.0006
t = 3	0.4405	t = 28	0.4497	t = 53	0.4330	t = 78	0.3943
t = 4	0.3609	t = 29	0.2178	t = 54	0.0733	t = 79	0.0646
t = 5	0.7192	t = 30	0.7762	t = 55	0.7995	t = 80	0.7878
t = 6	0.2258	t = 31	0.1078	t = 56	0.0262	t = 81	0.1694
t = 7	0.3175	t = 32	0.3773	t = 57	0.4223	t = 82	0.4468
t = 8	0.0127	t = 33	0.0001	t = 58	0.0085	t = 83	0.0372
t = 9	0.4290	t = 34	0.3877	t = 59	0.3303	t = 84	0.2632
t = 10	0.0544	t = 35	0.0821	t = 60	0.2037	t = 85	0.3048
t = 11	0.8000	t = 36	0.7836	t = 61	0.7332	t = 86	0.6516
t = 12	0.0450	t = 37	0.1887	t = 62	0.3328	t = 87	0.4690
t = 13	0.4268	t = 38	0.4483	t = 63	0.4445	t = 88	0.4132
t = 14	0.0112	t = 39	0.0424	t = 64	0.0909	t = 89	0.1523
t = 15	0.3218	t = 40	0.2539	t = 65	0.1838	t = 90	0.1182
t = 16	0.2185	t = 41	0.3164	t = 66	0.3888	t = 91	0.4334
t = 17	0.7240	t = 42	0.6386	t = 67	0.5277	t = 92	0.3978
t = 18	0.3516	t = 43	0.4862	t = 68	0.6042	t = 93	0.6987
t = 19	0.4420	t = 44	0.4068	t = 69	0.3435	t = 94	0.2538
t = 20	0.0984	t = 45	0.1611	t = 70	0.2304	t = 95	0.2998
t = 21	0.1747	t = 46	0.1101	t = 71	0.0568	t = 96	0.0195
t = 22	0.3964	t = 47	0.4372	t = 72	0.4500	t = 97	0.4366
t = 23	0.5114	t = 48	0.3795	t = 73	0.2371	t = 98	0.0924
t = 24	0.6183	t = 49	0.7092	t = 74	0.7705	t = 99	0.7984
t = 25	0.3330	t = 50	0.2400	t = 75	0.1246	 t = 100	0.0077
"""

raw_test = """
t = 101	0.4173
t = 102	0.0062
t = 103	0.3387
t = 104	0.1886
t = 105	0.7418
t = 106	0.3138
t = 107	0.4466
t = 108	0.0835
t = 109	0.1930
t = 110	0.3807
t = 111	0.5438
t = 112	0.5897
t = 113	0.3536
t = 114	0.2210
t = 115	0.0631
t = 116	0.4499
t = 117	0.2564
t = 118	0.7642
t = 119	0.1411
t = 120	0.3626
"""

train_data = []
for line in raw_train.strip().split("\n"):
    # Split by "t = "
    parts = line.split("t =")
    for p in parts:
        p = p.strip()
        if p:
            # Should be like "1   0.1701"
            subparts = re.split(r'\s+', p)
            if len(subparts) >= 2:
                train_data.append({
                    "t": int(subparts[0]),
                    "f": float(subparts[1])
                })

test_data = []
for line in raw_test.strip().split("\n"):
    parts = line.split("t =")
    for p in parts:
        p = p.strip()
        if p:
            subparts = re.split(r'\s+', p)
            if len(subparts) >= 2:
                test_data.append({
                    "t": int(subparts[0]),
                    "f": float(subparts[1])
                })

df_train = pd.DataFrame(train_data).sort_values("t")
df_train.to_csv("treinamento.csv", index=False)

df_test = pd.DataFrame(test_data).sort_values("t")
df_test.to_csv("teste.csv", index=False)

print(f"Saved {len(df_train)} train samples and {len(df_test)} test samples.")
