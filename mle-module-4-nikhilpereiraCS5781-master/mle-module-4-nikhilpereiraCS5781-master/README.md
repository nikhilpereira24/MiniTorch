# MiniTorch Module 4

<img src="https://minitorch.github.io/_images/match.png" width="100px">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments.

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py project/run_fast_tensor.py project/parallel_check.py tests/test_tensor_general.py

# MNIST Hidden States Image
![MNIST Image Hidden States](https://user-images.githubusercontent.com/89815451/145051336-a2167d21-f378-46a7-bbcc-26a3e6c60991.PNG)

# MNIST Training Log

Epoch: 1/500, loss: 2.3116411621676747, correct: 0
Epoch: 1/500, loss: 2.2946145380734393, correct: 0
Epoch: 1/500, loss: 2.3163151129754493, correct: 0
Epoch: 1/500, loss: 2.289229574772896, correct: 0
Epoch: 1/500, loss: 2.3291907755961163, correct: 1
Epoch: 1/500, loss: 2.3008059871389857, correct: 1
Epoch: 1/500, loss: 2.3092310877479845, correct: 2
Epoch: 1/500, loss: 2.2780577275593448, correct: 1
Epoch: 1/500, loss: 2.31269128565407, correct: 1
Epoch: 1/500, loss: 2.293356178793186, correct: 1
Epoch: 1/500, loss: 2.3062876440289535, correct: 1
Epoch: 1/500, loss: 2.3072422158458346, correct: 3
Epoch: 1/500, loss: 2.293578020685249, correct: 2
Epoch: 1/500, loss: 2.283497412379808, correct: 2
Epoch: 1/500, loss: 2.297651185130718, correct: 2
Epoch: 1/500, loss: 2.293810935711847, correct: 3
Epoch: 1/500, loss: 2.298521729348583, correct: 2
Epoch: 1/500, loss: 2.308155125035007, correct: 2
Epoch: 1/500, loss: 2.2805660621604797, correct: 2
Epoch: 1/500, loss: 2.30126000115634, correct: 2
Epoch: 1/500, loss: 2.2988974007933605, correct: 2
Epoch: 1/500, loss: 2.2925528935682014, correct: 2
Epoch: 1/500, loss: 2.2659005572874236, correct: 3
Epoch: 1/500, loss: 2.264984599068414, correct: 2
Epoch: 1/500, loss: 2.2933099233771572, correct: 2
Epoch: 1/500, loss: 2.281249240626649, correct: 2
Epoch: 1/500, loss: 2.2964404653900448, correct: 2
Epoch: 1/500, loss: 2.2736793138152307, correct: 2
Epoch: 1/500, loss: 2.2253691441646786, correct: 3
Epoch: 1/500, loss: 2.223780747880527, correct: 4
Epoch: 1/500, loss: 2.2579878029925675, correct: 2
Epoch: 1/500, loss: 2.3058097560403334, correct: 2
Epoch: 1/500, loss: 2.218934077118209, correct: 4
Epoch: 1/500, loss: 2.214885774969269, correct: 5
Epoch: 1/500, loss: 2.2622579047523708, correct: 4
Epoch: 1/500, loss: 2.1586428046569446, correct: 4
Epoch: 1/500, loss: 2.241525100648709, correct: 3
Epoch: 1/500, loss: 2.1929208955781636, correct: 4
Epoch: 1/500, loss: 2.1848557794341628, correct: 6
Epoch: 1/500, loss: 2.1806692720899146, correct: 7
Epoch: 1/500, loss: 2.1906540796653937, correct: 6
Epoch: 1/500, loss: 1.9960463195594096, correct: 4
Epoch: 1/500, loss: 2.070491123096269, correct: 6
Epoch: 1/500, loss: 1.9442849196667045, correct: 7
Epoch: 1/500, loss: 1.965236507076883, correct: 6
Epoch: 1/500, loss: 1.96145107821244, correct: 8
Epoch: 1/500, loss: 1.965136242073133, correct: 8
Epoch: 1/500, loss: 1.9337796530559703, correct: 5
Epoch: 1/500, loss: 1.7187785004898744, correct: 7
Epoch: 1/500, loss: 1.4886376425718095, correct: 9
Epoch: 1/500, loss: 1.5790983753120664, correct: 8
Epoch: 1/500, loss: 1.8119093211799615, correct: 8
Epoch: 1/500, loss: 1.9079438666350543, correct: 8
Epoch: 1/500, loss: 1.808373348983444, correct: 8
Epoch: 1/500, loss: 1.5072420319711615, correct: 7
Epoch: 1/500, loss: 1.68300941231701, correct: 10
Epoch: 1/500, loss: 1.3302072938478926, correct: 8
Epoch: 1/500, loss: 1.452806509868543, correct: 7
Epoch: 1/500, loss: 1.9658275159301146, correct: 10
Epoch: 1/500, loss: 1.3643947223751922, correct: 10
Epoch: 1/500, loss: 1.3375563061382303, correct: 10
Epoch: 1/500, loss: 1.5703368625961063, correct: 9
Epoch: 2/500, loss: 1.0992927214029684, correct: 10
Epoch: 2/500, loss: 1.9466774558680988, correct: 5
Epoch: 2/500, loss: 1.6639477186651035, correct: 9
Epoch: 2/500, loss: 1.1437281452881893, correct: 9
Epoch: 2/500, loss: 1.4045367492189378, correct: 10
Epoch: 2/500, loss: 1.0207579749222822, correct: 12
Epoch: 2/500, loss: 1.1307532098242479, correct: 11
Epoch: 2/500, loss: 0.6600472197985569, correct: 11
Epoch: 2/500, loss: 1.411475046353698, correct: 10
Epoch: 2/500, loss: 1.2857906839257542, correct: 14
Epoch: 2/500, loss: 1.1234236532157336, correct: 12
Epoch: 2/500, loss: 1.2319022938307929, correct: 8
Epoch: 2/500, loss: 2.065421464468878, correct: 7
Epoch: 2/500, loss: 1.3471170601775957, correct: 10
Epoch: 2/500, loss: 1.1018612086008637, correct: 13
Epoch: 2/500, loss: 1.0441206362507545, correct: 14
Epoch: 2/500, loss: 1.3258562881247253, correct: 14
Epoch: 2/500, loss: 0.8696065466867984, correct: 12
Epoch: 2/500, loss: 0.48216690503490917, correct: 14
Epoch: 2/500, loss: 0.9158777870959478, correct: 13
Epoch: 2/500, loss: 0.7072448638655878, correct: 14
Epoch: 2/500, loss: 0.8551267557086293, correct: 11
Epoch: 2/500, loss: 0.6646890906074201, correct: 11
Epoch: 2/500, loss: 0.9046449887642916, correct: 15
Epoch: 2/500, loss: 1.2583838425337655, correct: 15
Epoch: 2/500, loss: 1.032326197709031, correct: 14
Epoch: 2/500, loss: 0.6966606627531232, correct: 16
Epoch: 2/500, loss: 0.8796732191321419, correct: 15
Epoch: 2/500, loss: 0.6673961244569543, correct: 12
Epoch: 2/500, loss: 0.7904246699509977, correct: 11
Epoch: 2/500, loss: 1.5221539214305375, correct: 8
Epoch: 2/500, loss: 2.1873528385316865, correct: 10
Epoch: 2/500, loss: 0.9432552233313151, correct: 11
Epoch: 2/500, loss: 1.3531469324726961, correct: 13
Epoch: 2/500, loss: 1.167531788096709, correct: 13
Epoch: 2/500, loss: 0.6688303215626263, correct: 13
Epoch: 2/500, loss: 1.2208091141591153, correct: 11
Epoch: 2/500, loss: 0.8349969145468524, correct: 13
Epoch: 2/500, loss: 1.3634721878694231, correct: 11
Epoch: 2/500, loss: 1.1470991395223271, correct: 12
Epoch: 2/500, loss: 1.1842079492171884, correct: 12
Epoch: 2/500, loss: 1.2035640673735757, correct: 11
Epoch: 2/500, loss: 0.9538891059233248, correct: 12
Epoch: 2/500, loss: 0.6665607149304891, correct: 12
Epoch: 2/500, loss: 0.8579253136595559, correct: 12
Epoch: 2/500, loss: 0.6772096546172298, correct: 13
Epoch: 2/500, loss: 0.791563513705392, correct: 12
Epoch: 2/500, loss: 1.1521138987188175, correct: 10
Epoch: 2/500, loss: 1.2123574934240802, correct: 9
Epoch: 2/500, loss: 0.9373554193679117, correct: 14
Epoch: 2/500, loss: 0.95990073449622, correct: 13
Epoch: 2/500, loss: 1.2225620191261222, correct: 14
Epoch: 2/500, loss: 1.0865730364309443, correct: 12
Epoch: 2/500, loss: 1.0088650130530434, correct: 11
Epoch: 2/500, loss: 0.9399938611837194, correct: 11
Epoch: 2/500, loss: 0.8073009493501637, correct: 13
Epoch: 2/500, loss: 0.9305439839613936, correct: 12
Epoch: 2/500, loss: 1.0873394158061462, correct: 12
Epoch: 2/500, loss: 0.8045239839959061, correct: 13
Epoch: 2/500, loss: 0.73624672470075, correct: 13
Epoch: 2/500, loss: 0.8847271489513159, correct: 14
Epoch: 2/500, loss: 0.8339545478073549, correct: 11
Epoch: 3/500, loss: 0.3827271120728984, correct: 12
Epoch: 3/500, loss: 0.8792528332097109, correct: 13
Epoch: 3/500, loss: 0.7920468470997516, correct: 13
Epoch: 3/500, loss: 0.6959270169305192, correct: 11
Epoch: 3/500, loss: 0.520624395168342, correct: 11
Epoch: 3/500, loss: 0.8770933347140805, correct: 11
Epoch: 3/500, loss: 0.8244691785258969, correct: 11
Epoch: 3/500, loss: 0.2990528630844319, correct: 14
Epoch: 3/500, loss: 1.6730906508089878, correct: 12
Epoch: 3/500, loss: 0.9568892469624215, correct: 13
Epoch: 3/500, loss: 0.9484607487212481, correct: 15
Epoch: 3/500, loss: 0.9038464203010556, correct: 10
Epoch: 3/500, loss: 0.863794594083914, correct: 14
Epoch: 3/500, loss: 0.37138842166365055, correct: 13
Epoch: 3/500, loss: 1.1508584239878656, correct: 11
Epoch: 3/500, loss: 1.1657362560572742, correct: 12
Epoch: 3/500, loss: 1.1638710874450997, correct: 12
Epoch: 3/500, loss: 0.7371488376176433, correct: 12
Epoch: 3/500, loss: 0.34505318739139296, correct: 14
Epoch: 3/500, loss: 0.6361350905226458, correct: 13
Epoch: 3/500, loss: 0.5144801704145169, correct: 14
Epoch: 3/500, loss: 0.5478849447584822, correct: 12
Epoch: 3/500, loss: 0.5021089478911636, correct: 14
Epoch: 3/500, loss: 0.49356503810845437, correct: 14
Epoch: 3/500, loss: 0.8538149053486329, correct: 13
Epoch: 3/500, loss: 0.7836473476129482, correct: 13
Epoch: 3/500, loss: 0.7634024618285844, correct: 13
Epoch: 3/500, loss: 0.7468905271547224, correct: 14
Epoch: 3/500, loss: 0.4619283774749853, correct: 14
Epoch: 3/500, loss: 0.45491077950259545, correct: 12
Epoch: 3/500, loss: 0.8624060140666887, correct: 13
Epoch: 3/500, loss: 1.071466254705091, correct: 14
Epoch: 3/500, loss: 0.5669842663189095, correct: 14
Epoch: 3/500, loss: 1.3186254384540024, correct: 13
Epoch: 3/500, loss: 0.7692362660032122, correct: 13
Epoch: 3/500, loss: 0.4980470920542721, correct: 15
Epoch: 3/500, loss: 0.7995012937733746, correct: 11
Epoch: 3/500, loss: 0.6708682260197312, correct: 14
Epoch: 3/500, loss: 1.1007690510327726, correct: 13
Epoch: 3/500, loss: 1.2136834341158358, correct: 14
Epoch: 3/500, loss: 0.9725897728270366, correct: 12
Epoch: 3/500, loss: 0.7426774253157512, correct: 13
Epoch: 3/500, loss: 0.5738332820859922, correct: 12
Epoch: 3/500, loss: 0.3116243041996096, correct: 12
Epoch: 3/500, loss: 0.571223747327105, correct: 12
Epoch: 3/500, loss: 0.6434427666848054, correct: 13
Epoch: 3/500, loss: 0.8777406356915889, correct: 11
Epoch: 3/500, loss: 0.7907353469275589, correct: 13
Epoch: 3/500, loss: 0.8950638464835126, correct: 14
Epoch: 3/500, loss: 0.7255409823973609, correct: 13
Epoch: 3/500, loss: 0.5952964387436305, correct: 13
Epoch: 3/500, loss: 0.9973985771921092, correct: 16
Epoch: 3/500, loss: 0.6125408631434817, correct: 14
Epoch: 3/500, loss: 1.3752376591371818, correct: 12
Epoch: 3/500, loss: 0.8517335158405588, correct: 11
Epoch: 3/500, loss: 0.6977086357233955, correct: 12
Epoch: 3/500, loss: 0.8864181904028339, correct: 12
Epoch: 3/500, loss: 1.183760665713542, correct: 14
Epoch: 3/500, loss: 0.6195211012934625, correct: 15
Epoch: 3/500, loss: 0.4388410419576732, correct: 14
Epoch: 3/500, loss: 0.849989501470068, correct: 15
Epoch: 3/500, loss: 0.5353375424823369, correct: 13
Epoch: 4/500, loss: 0.39204394459925135, correct: 13
Epoch: 4/500, loss: 0.9940298878846667, correct: 12
Epoch: 4/500, loss: 1.0199272166409041, correct: 14
Epoch: 4/500, loss: 0.4860978103547257, correct: 11
Epoch: 4/500, loss: 0.4170756690153271, correct: 12
Epoch: 4/500, loss: 0.7187532252480382, correct: 12
Epoch: 4/500, loss: 0.5696123296387667, correct: 12
Epoch: 4/500, loss: 0.35330009515765015, correct: 15
Epoch: 4/500, loss: 1.1968019649232393, correct: 15
Epoch: 4/500, loss: 0.7608583273530867, correct: 14
Epoch: 4/500, loss: 0.9059602457288739, correct: 15
Epoch: 4/500, loss: 1.0075887433477049, correct: 12
Epoch: 4/500, loss: 0.6390813565975808, correct: 15
Epoch: 4/500, loss: 0.2935859460281656, correct: 16
Epoch: 4/500, loss: 0.9221334029119328, correct: 15
Epoch: 4/500, loss: 0.5780321489552714, correct: 13
Epoch: 4/500, loss: 0.9079778079072391, correct: 14
Epoch: 4/500, loss: 0.6243575957559289, correct: 13
Epoch: 4/500, loss: 0.3368817376652273, correct: 13
Epoch: 4/500, loss: 0.5216537384661755, correct: 13
Epoch: 4/500, loss: 0.27839133097962404, correct: 15
Epoch: 4/500, loss: 0.3624903699856477, correct: 15
Epoch: 4/500, loss: 0.43029934541183873, correct: 14
Epoch: 4/500, loss: 0.33247416688807246, correct: 14
Epoch: 4/500, loss: 0.3612598059018845, correct: 14
Epoch: 4/500, loss: 0.6026430608205464, correct: 13
Epoch: 4/500, loss: 0.3246372637664473, correct: 14
Epoch: 4/500, loss: 0.5393396696833075, correct: 15
Epoch: 4/500, loss: 0.6957988700369541, correct: 13
Epoch: 4/500, loss: 0.397442047053991, correct: 13
Epoch: 4/500, loss: 0.46845638068738793, correct: 13
Epoch: 4/500, loss: 1.3552831342391807, correct: 14
Epoch: 4/500, loss: 0.7117265684528132, correct: 13
Epoch: 4/500, loss: 1.0102217121451704, correct: 13
Epoch: 4/500, loss: 0.7969127880100976, correct: 14
Epoch: 4/500, loss: 0.4724277794199024, correct: 14
Epoch: 4/500, loss: 0.8079424158126755, correct: 12
Epoch: 4/500, loss: 0.5129126375404673, correct: 14
Epoch: 4/500, loss: 0.45188225830192646, correct: 14
Epoch: 4/500, loss: 1.1418102400489656, correct: 15
Epoch: 4/500, loss: 0.4615473749563506, correct: 15
Epoch: 4/500, loss: 0.6863605128225061, correct: 14
Epoch: 4/500, loss: 0.5386531824048116, correct: 11
Epoch: 4/500, loss: 0.442091496283178, correct: 14
Epoch: 4/500, loss: 0.689976462812018, correct: 13
Epoch: 4/500, loss: 0.3895768056945678, correct: 15
Epoch: 4/500, loss: 0.4840574606020867, correct: 15
Epoch: 4/500, loss: 0.5079660992357227, correct: 12
Epoch: 4/500, loss: 0.835609056060255, correct: 13
Epoch: 4/500, loss: 0.4096818945584506, correct: 13
Epoch: 4/500, loss: 0.48218824936077787, correct: 13
Epoch: 4/500, loss: 0.6103955793628908, correct: 15
Epoch: 4/500, loss: 0.9303495021548028, correct: 13
Epoch: 4/500, loss: 0.7099803161934531, correct: 14
Epoch: 4/500, loss: 0.6670011884329395, correct: 13
Epoch: 4/500, loss: 0.5106233527352839, correct: 15
Epoch: 4/500, loss: 0.8370604866063067, correct: 15
Epoch: 4/500, loss: 0.49305624167184164, correct: 15
Epoch: 4/500, loss: 0.7294574222102078, correct: 15
Epoch: 4/500, loss: 0.3813834247177039, correct: 14
Epoch: 4/500, loss: 0.672252994785257, correct: 15
Epoch: 4/500, loss: 0.5177584210157302, correct: 15
Epoch: 5/500, loss: 0.45503708339722204, correct: 15
Epoch: 5/500, loss: 1.1537343930570583, correct: 15
Epoch: 5/500, loss: 0.7760517316714388, correct: 16
Epoch: 5/500, loss: 0.4366695865700302, correct: 16
Epoch: 5/500, loss: 0.45884349822384474, correct: 14
Epoch: 5/500, loss: 0.35821287863584217, correct: 14
Epoch: 5/500, loss: 0.4649550264416466, correct: 14
Epoch: 5/500, loss: 0.21554180985427784, correct: 14
Epoch: 5/500, loss: 0.8913782367926767, correct: 13
Epoch: 5/500, loss: 0.5913491789495382, correct: 13
Epoch: 5/500, loss: 0.5705910660765163, correct: 14
Epoch: 5/500, loss: 0.6414857034805412, correct: 11
Epoch: 5/500, loss: 0.3418094662451859, correct: 13
Epoch: 5/500, loss: 0.28719682105551336, correct: 13
Epoch: 5/500, loss: 1.0616932983284737, correct: 15
Epoch: 5/500, loss: 0.5213823237794109, correct: 14
Epoch: 5/500, loss: 0.6297161428098637, correct: 14
Epoch: 5/500, loss: 0.3213916343636656, correct: 13
Epoch: 5/500, loss: 0.20904960792512192, correct: 12
Epoch: 5/500, loss: 0.3156608842468124, correct: 14
Epoch: 5/500, loss: 0.5175942191474827, correct: 14
Epoch: 5/500, loss: 0.3089408643048895, correct: 13
Epoch: 5/500, loss: 0.1268950555432658, correct: 14
Epoch: 5/500, loss: 0.913504564262654, correct: 14
Epoch: 5/500, loss: 1.0093779264271452, correct: 16
Epoch: 5/500, loss: 0.7334184793599055, correct: 15
Epoch: 5/500, loss: 0.3774565927498038, correct: 14
Epoch: 5/500, loss: 0.32525979681194306, correct: 15
Epoch: 5/500, loss: 0.4186478624027263, correct: 12
Epoch: 5/500, loss: 0.42798839755904194, correct: 13
Epoch: 5/500, loss: 0.5399653595023463, correct: 12
Epoch: 5/500, loss: 0.7386882612970578, correct: 13
Epoch: 5/500, loss: 0.4811309926832331, correct: 13
Epoch: 5/500, loss: 0.4171478026281718, correct: 14
Epoch: 5/500, loss: 0.28200914021032986, correct: 14
Epoch: 5/500, loss: 0.33052336116685493, correct: 15
Epoch: 5/500, loss: 0.5145594182993873, correct: 13
Epoch: 5/500, loss: 0.39349490636646267, correct: 14
Epoch: 5/500, loss: 0.328096248201252, correct: 13
Epoch: 5/500, loss: 1.3232473341822857, correct: 14
Epoch: 5/500, loss: 0.4765873385317174, correct: 14
Epoch: 5/500, loss: 0.4515869905274342, correct: 14
Epoch: 5/500, loss: 0.6290976937070195, correct: 14
Epoch: 5/500, loss: 0.19273768250909684, correct: 13
Epoch: 5/500, loss: 0.15652315234562308, correct: 13
Epoch: 5/500, loss: 0.3611090133456403, correct: 15
Epoch: 5/500, loss: 0.34379808789764044, correct: 16
Epoch: 5/500, loss: 0.44424043024275894, correct: 13
Epoch: 5/500, loss: 0.6692699043971754, correct: 14
Epoch: 5/500, loss: 0.453692673674018, correct: 15
Epoch: 5/500, loss: 0.4039163631698206, correct: 15
Epoch: 5/500, loss: 0.31072443029971114, correct: 15
Epoch: 5/500, loss: 0.4718185388929635, correct: 14
Epoch: 5/500, loss: 0.7110703886335207, correct: 14
Epoch: 5/500, loss: 0.4231533487653966, correct: 14
Epoch: 5/500, loss: 0.6991001325434498, correct: 15
Epoch: 5/500, loss: 0.9186082955465028, correct: 15
Epoch: 5/500, loss: 0.6534847352372133, correct: 14
Epoch: 5/500, loss: 0.6325386522812322, correct: 15
Epoch: 5/500, loss: 0.45832051029186355, correct: 14
Epoch: 5/500, loss: 0.8484917440359653, correct: 14
Epoch: 5/500, loss: 0.43121018787520937, correct: 14
Epoch: 6/500, loss: 0.2617504581265526, correct: 14
Epoch: 6/500, loss: 0.28001287160549293, correct: 14
Epoch: 6/500, loss: 0.4389200993844946, correct: 14
Epoch: 6/500, loss: 0.3417953638736457, correct: 13
Epoch: 6/500, loss: 0.5883786257566712, correct: 12
Epoch: 6/500, loss: 0.3905226609663759, correct: 12
Epoch: 6/500, loss: 0.6013030386515243, correct: 14
Epoch: 6/500, loss: 0.44440226476749767, correct: 14

# SST2 Training Log

Epoch 1, loss 31.474527495861995, train accuracy: 51.11%
Validation accuracy: 54.00%
Best Valid accuracy: 54.00%
Epoch 2, loss 31.03261784127966, train accuracy: 50.89%
Validation accuracy: 51.00%
Best Valid accuracy: 54.00%
Epoch 3, loss 30.900484309716735, train accuracy: 52.67%
Validation accuracy: 51.00%
Best Valid accuracy: 54.00%
Epoch 4, loss 30.655796020914853, train accuracy: 53.33%
Validation accuracy: 51.00%
Best Valid accuracy: 54.00%
Epoch 5, loss 30.563780925204405, train accuracy: 55.78%
Validation accuracy: 63.00%
Best Valid accuracy: 63.00%
Epoch 6, loss 30.380975690896882, train accuracy: 59.33%
Validation accuracy: 73.00%
Best Valid accuracy: 73.00%
Epoch 7, loss 30.137128127033254, train accuracy: 60.00%
Validation accuracy: 50.00%
Best Valid accuracy: 73.00%
Epoch 8, loss 29.712058349251762, train accuracy: 62.22%
Validation accuracy: 68.00%
Best Valid accuracy: 73.00%
Epoch 9, loss 29.678665900853428, train accuracy: 61.11%
Validation accuracy: 74.00%
Best Valid accuracy: 74.00%
Epoch 10, loss 29.051649393540256, train accuracy: 63.78%
Validation accuracy: 61.00%
Best Valid accuracy: 74.00%
Epoch 11, loss 28.640785431778724, train accuracy: 65.78%
Validation accuracy: 59.00%
Best Valid accuracy: 74.00%
Epoch 12, loss 28.365108161422164, train accuracy: 67.56%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 13, loss 27.903401699940552, train accuracy: 67.78%
Validation accuracy: 68.00%
Best Valid accuracy: 74.00%
Epoch 14, loss 27.19625841056963, train accuracy: 70.22%
Validation accuracy: 62.00%
Best Valid accuracy: 74.00%
Epoch 15, loss 26.86787025044872, train accuracy: 71.33%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 16, loss 26.55227732739782, train accuracy: 69.56%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 17, loss 25.723873714201133, train accuracy: 72.44%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 18, loss 25.027323752062212, train accuracy: 74.22%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 19, loss 24.274507458716904, train accuracy: 75.11%
Validation accuracy: 69.00%
Best Valid accuracy: 74.00%
Epoch 20, loss 24.203363310150696, train accuracy: 74.44%
Validation accuracy: 70.00%
Best Valid accuracy: 74.00%
Epoch 21, loss 22.79367464156934, train accuracy: 77.11%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 22, loss 22.576767041982475, train accuracy: 74.22%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 23, loss 21.720253491232707, train accuracy: 76.89%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 24, loss 21.499991179220313, train accuracy: 79.78%
Validation accuracy: 73.00%
Best Valid accuracy: 74.00%
Epoch 25, loss 21.193511594175988, train accuracy: 76.89%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 26, loss 20.62963187588892, train accuracy: 77.56%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 27, loss 20.381050464595386, train accuracy: 77.78%
Validation accuracy: 71.00%
Best Valid accuracy: 75.00%
Epoch 28, loss 19.925039719039923, train accuracy: 78.89%
Validation accuracy: 74.00%
Best Valid accuracy: 75.00%
Epoch 29, loss 18.944659669091347, train accuracy: 83.11%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 30, loss 18.23418912977125, train accuracy: 82.67%
Validation accuracy: 75.00%
Best Valid accuracy: 75.00%
Epoch 31, loss 17.687150533923433, train accuracy: 82.00%
Validation accuracy: 73.00%
Best Valid accuracy: 75.00%
Epoch 32, loss 17.411499002674688, train accuracy: 82.00%
Validation accuracy: 76.00%
Best Valid accuracy: 76.00%
