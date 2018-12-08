# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['BablTask1LoaderTest::test_converts_data_to_dataframe 1'] = '{"fact1":{"0":"John travelled to the hallway.","1":"Daniel went back to the bathroom.","2":"John went to the hallway.","3":"Sandra travelled to the hallway.","4":"Sandra went back to the bathroom."},"fact2":{"0":"Mary journeyed to the bathroom.","1":"John moved to the bedroom.","2":"Sandra journeyed to the kitchen.","3":"John went to the garden.","4":"Sandra moved to the kitchen."},"question":{"0":"Where is John? ","1":"Where is Mary? ","2":"Where is Sandra? ","3":"Where is Sandra? ","4":"Where is Sandra? "},"answer":{"0":"hallway","1":"bathroom","2":"kitchen","3":"hallway","4":"kitchen"}}'

snapshots['BablTask1LoaderTest::test_splits_dataframe_into_X_and_y 1'] = '{"fact1":{"0":"John travelled to the hallway.","1":"Daniel went back to the bathroom.","2":"John went to the hallway.","3":"Sandra travelled to the hallway.","4":"Sandra went back to the bathroom."},"fact2":{"0":"Mary journeyed to the bathroom.","1":"John moved to the bedroom.","2":"Sandra journeyed to the kitchen.","3":"John went to the garden.","4":"Sandra moved to the kitchen."},"question":{"0":"Where is John? ","1":"Where is Mary? ","2":"Where is Sandra? ","3":"Where is Sandra? ","4":"Where is Sandra? "}}'

snapshots['BablTask1LoaderTest::test_splits_dataframe_into_X_and_y 2'] = '{"answer":{"0":"hallway","1":"bathroom","2":"kitchen","3":"hallway","4":"kitchen"}}'
