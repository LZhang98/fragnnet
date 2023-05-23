from enum import Enum
'''
Just a lot of enum
'''
class Chorm(str, Enum):
	# chorm
	CE = 'CE'
	DI = 'DI'
	GC = 'GC'
	LC = 'LC'
	FI = 'FI'
	IC = 'IC'

class Ionlization(str, Enum):
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

class MsLevel(str, Enum):
	MS1 = 'MS1'
	MS2 = 'MS2'

class MsLevel(str, Enum):
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

class CIDType(str, Enum):
	HCD = 'hcd'
	CID = 'cid'

class AdductType(str, Enum):
	M_H_pos = '[M+H]+'
	M_H_neg = '[M-H]-'
	M_pos = '[M]+'

