"""

    Here we collect the various cross-mappings between datasets for open-set testing

"""

city19_to_idd17 =         [0, 1, 2, 3, 4, 5,  6,  7,  8, 10, 11, 12, 13, 14, 15, 17, 18]
city19_to_synthia16 =     [0, 1, 2, 3, 4, 5,  6,  7,  8, 10, 11, 12, 13, 15, 17, 18]
city19_to_idda16 =        [0, 1, 2, 3, 4, 5,  6,  7,  8,  9, 10, 11, 12, 13, 17, 18]
city19_to_sii15 =         [0, 1, 2, 3, 4, 5,  6,  7,  8, 10, 11, 12, 13, 17, 18]
city19_to_crosscity13 =   [0, 1, 2, 6, 7, 8,  9, 10, 11, 12, 13, 15, 17, 18]
city19_to_cci12 =         [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 17, 18]

idd17_to_synthia16 =      [0, 1, 2, 3, 4, 5,  6,  7,  8,  9, 10, 11, 12, 14, 15, 16]
idd17_to_sii15 =          [0, 1, 2, 3, 4, 5,  6,  7,  8,  9, 10, 11, 12, 15, 16]
idd17_to_crosscity13 =    [0, 1, 2, 6, 7, 8,  9, 10, 11, 12, 14, 15, 16]
idd17_to_cci12 =          [0, 1, 2, 6, 7, 8,  9, 10, 11, 12, 15, 16]

idda16_to_sii15 =         [0, 1, 2, 3, 4, 5,  6,  7,  8, 10, 11, 12, 13, 14, 15]
idda16_to_cci12 =         [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 14, 15]

synthia16_to_sii15 =      [0, 1, 2, 3, 4, 5,  6,  7,  8,  9, 10, 11, 12, 14, 15]
synthia16_to_crosscity13 =[0, 1, 2, 6, 7, 8,  9, 10, 11, 12, 13, 14, 15]
synthia16_to_cci12 =      [0, 1, 2, 6, 7, 8,  9, 10, 11, 12, 14, 15]

sii15_to_cci12 =          [0, 1, 2, 6, 7, 8,  9, 10, 11, 12, 13, 14]

ids_dict = {
            "city19":{'idd17':city19_to_idd17, 'synthia16':city19_to_synthia16, 'idda16':city19_to_idda16, 
                      'sii15':city19_to_sii15, 'crosscity13':city19_to_crosscity13, 'cci12':city19_to_cci12},
            "idd17":{'synthia16': idd17_to_synthia16, 'sii15': idd17_to_sii15, 'crosscity13': idd17_to_crosscity13, 'cci12': idd17_to_cci12},
            "idda16": {'sii15': idda16_to_sii15, 'cci12': idda16_to_cci12},
            "synthia16": {'sii15': synthia16_to_sii15, 'crosscity13': synthia16_to_crosscity13, 'cci12': synthia16_to_cci12},
            "sii15": {'cci12': synthia16_to_sii15}
           }