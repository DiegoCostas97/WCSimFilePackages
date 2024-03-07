import pandas as pd
import numpy as np

def digihits_info_to_df(file, nevents):

    npz = np.load(file, allow_pickle=True)

    event = npz['event_id']
    digi_hit_pmt      = npz['digi_hit_pmt']
    digi_hit_charge   = npz['digi_hit_charge']
    digi_hit_time     = npz['digi_hit_time']
    digi_hit_position = npz['digi_hit_position']
    digi_hit_trigger  = npz['digi_hit_trigger']

    digi_hit_truehit_parent_trackID  = npz['digi_hit_truehit_parent_trackID']
    digi_hit_truehit_times           = npz['digi_hit_truehit_times']
    digi_hit_truehit_creator         = npz['digi_hit_truehit_creator']

    data = [list((ievt,
                  ipmt,
                  ich,
                  itin,
                  itrig,
                  [i[0] for i in ipos],
                  [i[1] for i in ipos],
                  [i[2] for i in ipos])) for ievt, ipmt, ich, itin, itrig, ipos in zip(event,
                                                                                       digi_hit_pmt,
                                                                                       digi_hit_charge,
                                                                                       digi_hit_time,
                                                                                       digi_hit_trigger,
                                                                                       digi_hit_position)]


    df = pd.DataFrame(data, columns=['event_id',
                                     'A',
                                     'B',
                                     'C',
                                     'D',
                                     'E',
                                     'F',
                                     'G'])

    df = df.explode(list('ABCDEFG'))

    df = df.rename(columns={"A": "digi_hit_pmt", "B": "digi_hit_charge", 'C': 'digi_hit_time',
                            'D': 'digi_hit_trigger', 'E': 'digi_hit_x', 'F': 'digi_hit_y', 'G':'digi_hit_z'})

    # Real Hit Parent Adding to DF ############################################################################################
    digi_hit_truehit_parent_trackID = [[i for i in irhp] for irhp in digi_hit_truehit_parent_trackID]

    df_rhp = pd.DataFrame()
    df_rhp['event'] = range(nevents)

    df_temp = pd.DataFrame({'event_id': df_rhp['event'], 'digi_hit_truehit_parent_trackID': digi_hit_truehit_parent_trackID})
    df_temp_exploded = df_temp.explode('digi_hit_truehit_parent_trackID')

    df  = df.sort_values(by='event_id').reset_index(drop=True)
    df_temp_exploded = df_temp_exploded.sort_values(by='event_id').reset_index(drop=True)

    df_result = pd.concat([df, df_temp_exploded['digi_hit_truehit_parent_trackID']], axis=1)
    ###########################################################################################################################
    # Real Hit Creator Adding to DF
    digi_hit_truehit_creator = [[i for i in irhcp] for irhcp in digi_hit_truehit_creator]

    df_rhcp = pd.DataFrame()
    df_rhcp['event'] = range(nevents)

    df_temp = pd.DataFrame({'event_id': df_rhcp['event'], 'digi_hit_truehit_creator': digi_hit_truehit_creator})
    df_temp_exploded = df_temp.explode('digi_hit_truehit_creator')

    df  = df.sort_values(by='event_id').reset_index(drop=True)
    df_temp_exploded = df_temp_exploded.sort_values(by='event_id').reset_index(drop=True)

    df_result = pd.concat([df_result, df_temp_exploded['digi_hit_truehit_creator']], axis=1)
    #########################################################################################################################
    # Real Hit Times Adding to DF
    digi_hit_truehit_times = [[i for i in irht] for irht in digi_hit_truehit_times]

    df_rht = pd.DataFrame()
    df_rht['event'] = range(nevents)

    df_temp = pd.DataFrame({'event_id': df_rht['event'], 'digi_hit_truehit_times': digi_hit_truehit_times})
    df_temp_exploded = df_temp.explode('digi_hit_truehit_times')

    df  = df.sort_values(by='event_id').reset_index(drop=True)
    df_temp_exploded = df_temp_exploded.sort_values(by='event_id').reset_index(drop=True)

    df_result = pd.concat([df_result, df_temp_exploded['digi_hit_truehit_times']], axis=1)
    ###############################################################################################
    x = np.float64(df_result['digi_hit_x'].values)
    y = np.float64(df_result['digi_hit_y'].values)
    z = np.float64(df_result['digi_hit_z'].values)

    df_result['digi_hit_r'] = np.sqrt(x**2+y**2+z**2)
    df_result = df_result.dropna(axis=0)

    return  df_result


def truehits_info_to_df(file):
    npz = np.load(file, allow_pickle=True)

    event                   = npz['event_id']
    hit_parent              = npz['true_hit_parent']
    true_hit_pmt            = npz['true_hit_pmt']
    true_hit_time           = npz['true_hit_time']
    true_hit_start_time     = npz['true_hit_start_time']
    true_hit_pos            = npz['true_hit_pos']
    true_hit_start_pos      = npz['true_hit_start_pos']
    true_hit_creatorProcess = npz['true_hit_creator_process']

    data = [list((i,
                  thparent,
                  thp,
                  tht,
                  thst,
                  thpos[:, 0],
                  thpos[:, 1],
                  thpos[:, 2],
                  thsp[:, 0],
                  thsp[:, 1],
                  thsp[:, 2],
                  thcp)) for i, thparent, thp, tht, thst, thpos, thsp, thcp in zip(event,
                                                                                   hit_parent,
                                                                                   true_hit_pmt,
                                                                                   true_hit_time,
                                                                                   true_hit_start_time,
                                                                                   true_hit_pos,
                                                                                   true_hit_start_pos,
                                                                                   true_hit_creatorProcess)]

    df = pd.DataFrame(data, columns=['event_id',
                                     'J',
                                     'A',
                                     'B',
                                     'C',
                                     'D',
                                     'E',
                                     'F',
                                     'G',
                                     'H',
                                     'I',
                                     'K'])

    df = df.explode(list('JABCDEFGHIK'))

    df = df.rename(columns={"A": "true_hit_pmt", "B": "true_hit_time", 'C': 'true_hit_start_time',
                            'D': 'hit_x', 'E': 'hit_y', 'F': 'hit_z',
                            'G': 'hit_start_x', 'H': 'hit_start_y', 'I': 'hit_start_z',
                            'J': 'true_hit_parent', 'K': 'true_hit_creatorProcess'})

    x = np.float64(df['hit_x'].values)
    y = np.float64(df['hit_z'].values)
    z = np.float64(df['hit_y'].values)

    x_start = np.float64(df['hit_start_x'].values)
    y_start = np.float64(df['hit_start_z'].values)
    z_start = np.float64(df['hit_start_y'].values)

    df['hit_r']       = np.sqrt(x**2+y**2+z**2)
    df['hit_start_r'] = np.sqrt(x_start**2+y_start**2+z_start**2)

    return df


def track_info_to_df(file):
    npz = np.load(file, allow_pickle=True)

    event                 = npz['event_id']
    position              = npz['position']
    direction             = npz['direction']
    track_id              = npz['track_id']
    track_pid             = npz['track_pid']
    track_parent          = npz['track_parent']
    track_creator_process = npz['track_creator_process']
    track_start_time      = npz['track_start_time']
    track_energy          = npz['track_energy']
    track_start_position  = npz['track_start_position']
    track_stop_position   = npz['track_stop_position']
    

    data = [list((i,
                  k[0],
                  k[1],
                  k[2],
                  l[0],
                  l[1],
                  l[2],
                  tpi,
                  ti,
                  tp,
                  tcp,
                  tst,
                  te,
                  tsp[:, 0],
                  tsp[:, 1],
                  tsp[:, 2],
                  top[:, 0],
                  top[:, 1],
                  top[:, 2])) for i, k, l, tpi, ti, tp, tcp, tst, te, tsp, top in zip(event,
                                                                         position,
                                                                         direction,
                                                                         track_pid,
                                                                         track_id,
                                                                         track_parent,
                                                                         track_creator_process,
                                                                         track_start_time,
                                                                         track_energy,
                                                                         track_start_position,
                                                                         track_stop_position)]

    df = pd.DataFrame(data, columns=['event_id',
                                     'xi',
                                     'yi',
                                     'zi',
                                     'dxi',
                                     'dyi',
                                     'dzi',
                                     'A',
                                     'J',
                                     'K',
                                     'L',
                                     'B',
                                     'C',
                                     'D',
                                     'E',
                                     'F',
                                     'G',
                                     'H',
                                     'I'])

    df = df.explode(list('AJKLBCDEFGHI'))

    df = df.rename(columns={"A": "track_pid", "J": "track_id", "K":"track_parent", "L":"track_creator_process",
                            "B": "track_ti", 'C': 'track_energy',
                            'D': 'track_xi', 'E': 'track_yi', 'F': 'track_zi',
                            'G': 'track_xf', 'H': 'track_yf', 'I': 'track_zf'})

    return df

def simple_track_info_to_df(file):
    npz = np.load(file, allow_pickle=True)

    event                 = npz['event_id']
    track_id              = npz['track_id']
    track_pid             = npz['track_pid']
    track_parent          = npz['track_parent']
    track_creator_process = npz['track_creator_process']
    track_energy          = npz['track_energy']
    track_start_position  = npz['track_start_position']
    track_stop_position   = npz['track_stop_position']
    
    data = [list((i,
                  tpi,
                  ti,
                  tp,
                  tcp,
                  te,
                  np.sqrt(tsp[:, 0]**2 + tsp[:, 1]**2 + tsp[:, 2]**2),
                  np.sqrt(top[:, 0]**2 + top[:, 1]**2 + top[:, 2]**2))) for i, tpi, ti, tp, tcp, te, tsp, top in zip(event,
                                                                                                           track_pid,
                                                                                                           track_id,
                                                                                                           track_parent,
                                                                                                           track_creator_process,
                                                                                                           track_energy,
                                                                                                           track_start_position,
                                                                                                           track_stop_position)]
    
    
    df = pd.DataFrame(data, columns=['event_id',
                                     'A',
                                     'J',
                                     'K',
                                     'L',
                                     'M',
                                     'N',
                                     'P'])

    df = df.explode(list('AJKLMNP'))

    df = df.rename(columns={"A": "track_pid", "J": "track_id", "K":"track_parent", "L":"track_creator_process",
                            "M": "track_energy", "N": "track_ri", "P": "track_rf"})

    return df
