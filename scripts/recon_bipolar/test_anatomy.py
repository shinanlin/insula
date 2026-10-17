"""Scientific edge cases for the recon-only adapter, executed on Slurm."""
import unittest
import numpy as np
from run import merge_label, make_pairs, load_pipeline, UPSTREAM


class AnatomyTests(unittest.TestCase):
    def test_atlas_boundary_and_unresolved(self):
        self.assertEqual(merge_label('ctx_lh_G_front_middle', 'insula anterior pole L'),
                         ('ctx_lh_G_front_middle', 'aparc2009s', 'hammers_insula_outside_aparc'))
        self.assertEqual(merge_label('ctx_lh_G_insular_short', 'insula posterior short gyrus L'),
                         ('insula posterior short gyrus L', 'hammers', ''))
        for h in ('0', 'insula anterior pole R', 'FL middle frontal gyrus L'):
            self.assertEqual(merge_label('ctx_lh_G_Ins_lg_and_S_cent_ins', h)[2],
                             'insula_subdivision_unresolved')

    def test_pairing_units_long_names_and_geometry_break(self):
        contacts = [dict(contact=f'RFAI-ANT{n}', shaft='RFAI-ANT', contact_number=n,
                         x=x, y=1., z=1., contact_type='D')
                    for n, x in ((10, 1.), (11, 4.), (12, 7.), (13, 40.))]
        result = make_pairs('D0157', contacts)
        self.assertEqual(result.pairs, [('RFAI-ANT10', 'RFAI-ANT11'), ('RFAI-ANT11', 'RFAI-ANT12')])
        np.testing.assert_allclose([r['dist_mm'] for r in result.pair_rows], [3., 3.])
        self.assertEqual(result.skipped_breaks[0]['reason'], 'distance')
        self.assertEqual(result.orphans[0]['contact'], 'RFAI-ANT13')

    def test_consensus_is_not_all_three_aic(self):
        pipeline = load_pipeline(UPSTREAM / 'common/parcellation.py')
        mapping = {'insula anterior pole L': 'AIC(L)', 'ctx_lh_G_front_middle': 'MFG(L)'}
        self.assertEqual(pipeline.consensus_roi(['0', 'insula anterior pole L', 'Left-Cerebral-White-Matter'], mapping), ('AIC', False))
        self.assertEqual(pipeline.consensus_roi(['insula anterior pole L', '0', 'ctx_lh_G_front_middle'], mapping), ('AIC–MFG', True))


if __name__ == '__main__':
    unittest.main()
