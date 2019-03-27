#This is a case where (A_1 kron A_2)^+ = A_1^+ kron A_2^+ fails.
import numpy as np

As_fail = [np.array([[ 1.35134806,  0.32022911, -0.66384512, -0.00773205],
       [ 1.35134806,  0.32022911, -0.66384512, -0.00773205],
       [ 0.09840087,  0.1308376 ,  0.25744384,  0.51331768],
       [-0.18297221,  1.35042614,  0.28422141, -0.45167534]]), np.array([[ 0.33038563,  0.58656473,  0.85167952, -0.76862987],
       [ 0.33038563,  0.58656473,  0.85167952, -0.76862987],
       [-0.39626097,  0.1245063 ,  0.63901164,  0.63274303],
       [ 0.83660162, -0.4454058 ,  0.27945482,  0.32934937]]), np.array([[ 0.05001878,  0.02009631,  0.15520316,  0.77468174],
       [ 0.05001878,  0.02009631,  0.15520316,  0.77468174],
       [ 0.23684572,  0.65637933,  0.17394329, -0.06716834],
       [ 1.72533359, -0.63334699,  0.00373024, -0.09571684]]), np.array([[ -2.47253205,   2.60057724,   0.68177092,   0.19018388],
       [ -2.47253205,   2.60057724,   0.68177092,   0.19018388],
       [  0.25177033,   0.11621459,   0.40702073,   0.22499435],
       [  5.22099892,   5.43495205,   1.24389123, -10.8998422 ]])]

Y_fail = np.array([8.74665836e-01, 6.68168182e-01, 7.28269416e-01, 6.45109535e-01
, 7.67602345e-01, 1.67907504e-01, 4.31941772e-01, 1.06382171e-01
, 5.60237457e-01, 4.52336252e-01, 9.26318633e-01, 5.87338680e-02
, 8.27862578e-01, 5.35584699e-01, 4.88178158e-01, 7.33226629e-01
, 7.31747325e-01, 8.93300979e-02, 7.85210790e-01, 7.45644188e-01
, 5.56466010e-01, 1.88653292e-01, 3.37730389e-02, 2.10433403e-01
, 9.71617809e-01, 3.80637148e-01, 8.45274400e-01, 1.00194682e-04
, 6.07665202e-01, 6.85077147e-01, 3.91204087e-02, 2.12165749e-01
, 4.38725602e-02, 6.85920552e-01, 1.98598062e-01, 6.12149986e-01
, 4.56696558e-03, 1.61560438e-01, 1.83273599e-01, 7.59330492e-01
, 7.25127869e-01, 8.55260831e-01, 3.00197016e-01, 5.19537900e-01
, 9.95048909e-01, 9.44077462e-01, 7.18815448e-02, 2.77842578e-01
, 7.96430021e-03, 1.98575139e-01, 2.32631323e-01, 2.41660339e-01
, 8.99445354e-01, 9.47609537e-01, 8.34673523e-01, 7.28931594e-01
, 3.30589622e-02, 7.74180595e-01, 2.28002569e-01, 2.56055812e-01
, 2.12304737e-02, 2.97160973e-01, 1.08971487e-02, 9.10981801e-01
, 9.74661722e-01, 5.66374498e-01, 9.03663368e-01, 1.23329896e-01
, 5.38853965e-02, 1.14515517e-01, 5.60582096e-01, 2.49515363e-01
, 4.48616057e-02, 4.54810456e-01, 3.91737564e-01, 2.48112953e-01
, 6.06833408e-02, 2.49638286e-01, 6.91162697e-01, 3.82979102e-01
, 6.82824263e-01, 2.48994029e-01, 1.85726017e-01, 1.63712882e-02
, 5.16141108e-01, 6.30972036e-01, 6.90284178e-01, 4.16560917e-01
, 7.30238792e-01, 5.29601552e-01, 6.70404831e-01, 7.60154043e-01
, 9.16147859e-01, 3.80939970e-01, 9.41045785e-01, 9.11684085e-01
, 6.04974681e-01, 4.06293172e-01, 4.40489221e-01, 3.09432899e-01
, 1.55161175e-01, 6.99529737e-01, 3.32234094e-01, 7.59123362e-01
, 9.02800678e-01, 9.86293733e-01, 9.31695769e-01, 5.69451167e-01
, 4.55496053e-01, 5.80194026e-01, 9.67168854e-01, 3.51528654e-01
, 6.78903789e-02, 1.77730571e-01, 6.30331865e-01, 7.73713564e-01
, 4.97667118e-01, 9.98000066e-01, 9.35506707e-01, 6.03642041e-01
, 2.30749758e-01, 4.44459654e-01, 8.59520887e-01, 1.78202548e-01
, 2.09853471e-01, 8.02284967e-01, 8.12984503e-01, 4.16972860e-01
, 5.59999015e-01, 8.46611757e-01, 7.78001951e-01, 7.12090277e-01
, 2.27702661e-01, 9.77561852e-01, 8.17685469e-01, 2.61016857e-01
, 4.38023732e-01, 5.11497804e-01, 9.09861393e-01, 6.20647114e-01
, 8.96257671e-01, 5.18556095e-01, 5.12201342e-01, 8.81616395e-01
, 2.18271704e-01, 2.12987157e-01, 3.49119744e-01, 4.15759546e-01
, 9.56519489e-01, 8.26206496e-01, 3.99860951e-01, 1.18826135e-01
, 5.20940641e-01, 9.41914778e-01, 1.22831629e-01, 8.32623656e-01
, 2.73643009e-01, 3.52955199e-01, 7.73809155e-01, 6.15376910e-01
, 6.01805292e-01, 9.14851184e-01, 4.79043407e-01, 7.07464667e-01
, 7.12613469e-01, 4.35150741e-01, 6.50534887e-01, 4.29030167e-01
, 7.24063685e-01, 6.57965807e-01, 8.10504979e-01, 1.42796274e-01
, 1.35178380e-01, 5.86608807e-01, 1.92529982e-01, 7.71801873e-01
, 7.78364026e-01, 3.41233948e-01, 6.28394585e-01, 8.63088416e-01
, 5.71009091e-01, 5.24762912e-01, 7.62662050e-01, 7.91187333e-01
, 2.47610356e-01, 1.03487720e-01, 8.99173045e-01, 6.35182219e-01
, 7.55073122e-01, 5.47883284e-01, 2.30685551e-01, 8.61216103e-01
, 1.07990954e-01, 9.07664295e-01, 9.68375303e-01, 3.27473972e-01
, 8.29959119e-01, 4.89799084e-01, 2.40787023e-01, 5.19796612e-01
, 4.88643323e-01, 5.92816984e-01, 7.53868865e-01, 9.48120587e-01
, 5.94349168e-03, 1.30513277e-01, 5.16333944e-01, 5.96651656e-01
, 1.94650958e-01, 9.20493300e-01, 1.35468293e-01, 8.51563497e-01
, 5.38629484e-01, 8.42058191e-01, 6.06767721e-02, 4.89347830e-01
, 4.37463318e-01, 9.32718804e-01, 8.92160185e-01, 8.32629852e-02
, 5.00078031e-01, 9.20234348e-01, 3.41117222e-01, 7.93480505e-01
, 8.85165583e-01, 2.33563369e-01, 4.25478549e-01, 7.48745837e-01
, 9.01992923e-01, 7.82790464e-01, 2.19521882e-01, 3.24960608e-01
, 9.05607442e-01, 7.91262213e-02, 9.90456328e-01, 5.83518147e-01
, 4.22111043e-01, 3.37294724e-01, 9.70521984e-01, 9.01316681e-01
, 5.34829860e-01, 2.72643192e-01, 4.51768953e-01, 1.93739225e-01
, 5.70841128e-01, 3.73571146e-02, 6.15980948e-01, 3.90914620e-01
, 7.32158488e-01, 8.40427483e-01, 3.12084672e-01, 5.18137944e-02
, 9.85134369e-01, 4.35738576e-01, 9.69096826e-01, 5.43650888e-01])
