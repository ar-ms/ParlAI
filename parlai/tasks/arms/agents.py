# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
"""PersonaChat task agents.

Persona can be 'none', 'self', 'other', or 'both'.
Format of persona can be 'original' or 'revised'.

--task arms:{TEACHER_NAME}:no_cands

This is specified in the following way:
--task personachat:{format}
...where {format} is one of...
- none
- self_original
- self_revised
- other_original
- other_revised
- both_original
- both_revised
"""


from parlai.core.teachers import FbDialogTeacher
from .build import build

import copy
import os

def _path(opt, persona, use_cands=False):
    # Build the data if it doesn't exist.
    build(opt)
    datatype =  opt['datatype'].split(':')[0]
    if datatype == 'test':
        print("WARNING: Test set not included. Setting datatype to valid.")
        datatype = 'valid'
    dt = datatype + '_' + persona
    cands = '' if use_cands else '_no_cands'
    return os.path.join(opt['datapath'], 'ConvAI2', dt + cands + '.txt')

class NoneTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'none_original')
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'none_original', use_cands)
        super().__init__(opt, shared)

class SelfOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'self_original')
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'self_original', use_cands)
        super().__init__(opt, shared)

class SelfTeacher(SelfOriginalTeacher):
    pass

class SelfRevisedTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'self_revised')
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'self_revised', use_cands)
        super().__init__(opt, shared)

class OtherOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'other_original')
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'other_original', use_cands)
        super().__init__(opt, shared)

class OtherTeacher(OtherOriginalTeacher):
    pass

class OtherRevisedTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'other_revised')
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'other_revised', use_cands)
        super().__init__(opt, shared)

class BothOriginalTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'both_original')
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'both_original', use_cands)
        super().__init__(opt, shared)

class BothTeacher(BothOriginalTeacher):
    pass

class BothRevisedTeacher(FbDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, 'both_revised')
        super().__init__(opt, shared)
        opt = copy.deepcopy(opt)
        try:
            cands = opt['task'].split(":")[2]
            use_cands = False if cands == 'no_cands' else True
        except:
            use_cands = True
        opt['datafile'] = _path(opt, 'both_revised', use_cands)
        super().__init__(opt, shared)

class DefaultTeacher(SelfOriginalTeacher):
    pass
