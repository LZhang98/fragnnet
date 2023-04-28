from enum import Enum
'''
Just a lot of enum
'''
class Chorm(Enum, str):
    # chorm
    CE = 'CE'
    DI = 'DI'
    GC = 'GC'
    LC = 'LC'
    FI = 'FI'
    IC = 'IC'

class Ionlization(Enum, str):
    # ionlization
    APCI = 'APCI'
    APPI = 'APPI'
    CI = 'CI'
    EI = 'EI'
    ESI = 'ESI'
    FAB = 'FAB'
    LSIMS = 'LSIMS'
    In_source_CID = 'In-source CID'
    PI = 'PI'

class MsLevel(Enum, str):
    MS1 = 'MS1'
    MS2 = 'MS2'

class MsLevel(Enum, str):
    QqQ = 'QqQ'
    QTOF = 'QTOF'
    BEqQ = 'BEqQ'
    Orbitrap = 'Orbitrap'
    QqLIT = 'QqLIT'
    API = 'API'
    ICR = 'ICR'
    EBEqEBE = 'EBEqEBE'
    EBE = 'EBE'
    BE = 'BE'
    IT = 'IT'
    QFT = 'QFT '  # Q Exactive Plus Orbitrap

    Pos = 'P'
    Neg = 'N'

class CIDType(Enum, str):
    HCD = 'hcd'
    CID = 'cid'

class AdductType(Enum, str):
    M_H_pos = '[M+H]+'
    M_H_neg = '[M-H]-'
    M_pos = '[M]+'

