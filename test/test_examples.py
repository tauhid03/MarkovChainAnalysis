#! /usr/bin/env python

import unittest
from examples.MDP_models import *
from random import randint
import numpy as np


class TestExamples(unittest.TestCase):

    # add global stuff here
    def setUp(self):
        return

    def test_neighbors(self):
        self.assertSequenceEqual(neighbors(0, env1), [4])
        self.assertSequenceEqual(neighbors(10, env1), [5,14,9])

    def test_envs(self):
        self.assertTrue(check_grid_topo(env1))
        self.assertTrue(check_grid_topo(env2))

    def test_simple_step_gen(self):
        ns = neighbors(3, env2)
        self.assertSequenceEqual(ns, [5,2])
        p1 = gridworld_step_prob(3,5, types2[0], env2)
        self.assertEqual(p1, 0.2)
        p2 = gridworld_step_prob(3,2, types2[1], env2)
        self.assertEqual(p2, 0.3)

    def test_transitions(self):
        output = np.array([[[[0.33333333, 0.33333333, 0.        , 0.        , 0.33333333, 0.  ],
                           [0.25      , 0.25      , 0.25      , 0.        , 0.25 , 0.      ],
                           [0.        , 0.33333333, 0.33333333, 0.33333333, 0. , 0.        ],
                           [0. , 0.  , 0.33333333, 0.33333333, 0.        , 0.33333333],
                           [0.33333333, 0.33333333, 0.        , 0.        , 0.33333333, 0.        ],
                           [0.        , 0.        , 0.        , 0.5       , 0.        , 0.5       ]]],
                       [[[0.29411765, 0.35294118, 0. , 0.        , 0.35294118, 0. ],
                           [0.26086957, 0.2173913 , 0.26086957, 0.        , 0.26086957, 0.        ],
                           [0.        , 0.35294118, 0.29411765, 0.35294118, 0.        , 0.        ],
                           [0.        , 0.        , 0.35294118, 0.29411765, 0.        , 0.35294118],
                           [0.5       , 0.08333333, 0.        , 0.        , 0.41666667, 0.        ],
                           [0.  , 0.        , 0.        , 0.16666667, 0.        , 0.83333333]]]])
        calcd =  mkTransitions(env2, types2, 1, len(env2), gridworld_step_prob)
        np.testing.assert_almost_equal(calcd, output, decimal=7, verbose=True)

    def test_encode_decode(self):
        N = randint(2,10)
        X = randint(5,10)
        states = np.random.choice(np.arange(X, dtype=np.uint), N)
        state_int = encodeJointState(states, X)
        decoded = decodeJointState(state_int, N, X)
        np.testing.assert_almost_equal(states, decoded, decimal=7, verbose=True)


