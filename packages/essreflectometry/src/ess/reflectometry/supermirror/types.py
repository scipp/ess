from typing import NewType

import scipp as sc

# TODO Better specification of what the supermirror parameters mean
MValue = NewType('MValue', sc.Variable)
''':math:`M` value of the supermirror'''
CriticalEdge = NewType('CriticalEdge', sc.Variable)
'''Critical edge value of the supermirror'''
Alpha = NewType('Alpha', sc.Variable)
''':math:`\\alpha` value of the supermirror'''
