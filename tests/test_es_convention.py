"""ES regression checks without retraining models or requiring research datasets.

Run: python -m unittest discover -s tests -p 'test_es_convention.py' -v
Dependencies: numpy, pandas, scipy, pyarrow.
"""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import norm, t as student_t

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('2ALL_models', '3Compare_model', '4Pricing_Test')
BOOKS = {name: json.loads((ROOT / f'{name}.ipynb').read_text()) for name in NAMES}
ALPHAS = (.01, .05, .10)


def source(book, cell):
    return ''.join(BOOKS[book]['cells'][cell]['source'])


def functions(book, cell, env):
    nodes = [node for node in ast.parse(source(book, cell)).body
             if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), book, 'exec'), env)


def environment():
    return dict(np=np, pd=pd, json=json, norm=norm, student_t=student_t,
                warnings=warnings, ALPHAS=ALPHAS,
                ALPHA_SUFFIX={a: f'{a * 100:g}' for a in ALPHAS})


def forecast(pi, mu, sigma, nu=None):
    row = dict(permno=1, mthcaldt=pd.Timestamp('2020-01-31'), model_id='test',
               target_ret_final=-.1, pi_vec=pi, mu_vec=mu, sigma_vec=sigma)
    if nu is not None:
        row['nu_vec'] = nu
    return pd.DataFrame([row])


class ESConventionTests(unittest.TestCase):
    def setUp(self):
        self.env = environment()
        functions(NAMES[0], 8, self.env)
        functions(NAMES[0], 21, self.env)
        functions(NAMES[0], 38, self.env)
        functions(NAMES[0], 39, self.env)

    def test_notebook_code_compiles(self):
        for name, book in BOOKS.items():
            self.assertEqual(book['nbformat'], 4)
            for i, cell in enumerate(book['cells']):
                if cell['cell_type'] == 'code':
                    code = '\n'.join(
                        line[:len(line)-len(line.lstrip())] + 'pass'
                        if line.lstrip().startswith(('!', '%')) else line
                        for line in source(name, i).splitlines())
                    compile(code, f'{name}:cell{i}', 'exec')

    def test_normal_matches_tail_integral_and_positive_tail_is_preserved(self):
        for mean in (0., 1.):
            for alpha in ALPHAS:
                q, es = self.env['normal_var_and_es'](alpha, mean, .1)
                expected = quad(lambda r: r * norm.pdf(r, mean, .1), -np.inf, q)[0] / alpha
                self.assertAlmostEqual(es, expected, places=9)
                self.assertLessEqual(es, q)
                self.assertEqual(es > 0, mean > 0)

    def test_mixtures_match_numerical_tail_integrals(self):
        for pi, mu, sigma, nu in [
            ([1.], [0.], [.1], None),
            ([.2, .5, .3], [-.2, .01, .1], [.15, .06, .1], None),
            ([1.], [0.], [.1], [5.]),
            ([.4, .6], [-.1, .03], [.1, .05], [4., 8.]),
        ]:
            frame = forecast(pi, mu, sigma, nu)
            risk = self.env['calculate_es_table'](frame)
            for alpha in ALPHAS:
                suffix = f'{alpha * 100:g}'
                q, es = risk.loc[0, [f'var_{suffix}', f'es_{suffix}']]
                def density(r):
                    return sum(p * (norm.pdf(r, m, s) if nu is None else
                                    student_t.pdf((r-m)/s, nu[k])/s)
                               for k, (p, m, s) in enumerate(zip(pi, mu, sigma)))
                expected = quad(lambda r: r * density(r), -np.inf, q)[0] / alpha
                self.assertAlmostEqual(es, expected, places=8)
                self.assertLessEqual(es, q)
            once = self.env['attach_var_es'](frame)
            twice = self.env['attach_var_es'](once)
            pd.testing.assert_frame_equal(once, twice)
            self.assertEqual(twice.attrs['es_convention'], 'return')

    def test_quantile_es_and_fz0(self):
        # Constant quantiles are a degenerate distribution: ES equals that return.
        self.env['QUANTILE_LEVELS'] = np.array([.001, .01, .05, .1, .5, .99])
        quantiles = np.repeat([[-.3], [.2]], 6, axis=1)
        result = self.env['add_quantile_distribution_results'](pd.DataFrame(index=range(2)), quantiles)
        for alpha in ALPHAS:
            np.testing.assert_allclose(result[f'es_{alpha * 100:g}'], [-.3, .2])
        functions(NAMES[1], 7, self.env)
        y, q, e = np.array([-.3, .1]), np.array([-.2, -.2]), np.array([-.25, -.25])
        score = self.env['fz0_score'](y, q, e, .05, np.array([True, True]))
        expected = -(y <= q).astype(float) * (q-y) / (.05*e) + q/e + np.log(-e) - 1
        np.testing.assert_allclose(score, expected)
        self.assertTrue(np.isfinite(score).all())

    def test_component_decomposition(self):
        env = self.env
        frame = forecast([.2, .5, .3], [-.2, .01, .1], [.15, .06, .1])
        frame = env['attach_var_es'](frame)
        frame['window'] = 0
        for k, label in enumerate(['adverse_component', 'normal_component', 'favorable_component']):
            frame[label] = k
        env.update(structured_raw=frame, key_cols=['permno', 'mthcaldt'],
                   label_cols=['adverse_component', 'normal_component', 'favorable_component'],
                   component_names=['adverse', 'normal', 'favorable'])
        functions(NAMES[1], 7, env)
        # Execute the actual decomposition, excluding only serialization/display.
        exec(source(NAMES[1], 18).split('structured_components.to_parquet(')[0], env)
        np.testing.assert_allclose(env['es_contribution'].sum(axis=1), frame['es_5'])
        np.testing.assert_allclose(env['severity'] * env['tail_weight'], env['es_contribution'])
        np.testing.assert_allclose(env['tail_weight'].sum(axis=1), 1., atol=1e-10)
        self.assertTrue((env['severity'] < 0).all())

    def test_legacy_and_new_parquet_are_equivalent_and_idempotent(self):
        for name in NAMES[1:]:
            env = environment()
            functions(name, 1, env)
            read = env['read_return_es_parquet']
            legacy = pd.DataFrame(dict(es_5=[.3], var_5=[-.2],
                es_contribution_adverse_5=[.2], severity_adverse_5=[.4], tail_weight_adverse_5=[.5]))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'old.parquet'
                legacy.to_parquet(path, index=False)
                with warnings.catch_warnings(record=True) as caught:
                    converted = read(path)
                self.assertTrue(caught)
                self.assertEqual(converted.es_5.iloc[0], -.3)
                self.assertEqual(converted.severity_adverse_5.iloc[0], -.4)
                self.assertEqual(converted.tail_weight_adverse_5.iloc[0], .5)
                new = Path(folder)/'new.parquet'
                converted.to_parquet(new, index=False)
                pd.testing.assert_frame_equal(read(new), converted)
                self.assertEqual(read(new, columns=['es_5']).es_5.iloc[0], -.3)
                pd.testing.assert_frame_equal(pd.read_parquet(path), legacy)
                converted.attrs['es_convention'] = 'unknown'
                converted.to_parquet(new, index=False)
                with self.assertRaises(ValueError):
                    read(new)

    def test_portfolio_and_regression_directions(self):
        env = self.env
        env.update(me_breakpoints=None, control_cols=[], industry_type='industry')
        for cell in (10, 11, 12, 30):
            functions(NAMES[2], cell, env)
        n = 100
        signal = np.linspace(-.5, -.05, n)
        panel = pd.DataFrame(dict(permno=np.arange(n), mthcaldt=pd.Timestamp('2020-01-31'),
            return_date=pd.Timestamp('2020-02-29'), signal=signal, target_ret_final=-.1*signal,
            mthcap=1., mthcap_log=0., bm=1., is_nyse=True, industry=1))
        portfolios = env['sort_portfolios'](panel, 'signal')
        self.assertLess(portfolios.mean_signal.iloc[0], portfolios.mean_signal.iloc[-1])
        hml = env['build_hml'](portfolios, {})
        self.assertTrue((hml.hml_return < 0).all())
        slope = env['run_fama_macbeth'](panel, 'signal', 'test', 'test', 'ES')
        self.assertTrue((slope.signal_slope < 0).all())
        flipped = panel.assign(signal=-panel.signal)
        reverse = env['build_hml'](env['sort_portfolios'](flipped, 'signal'), {})
        np.testing.assert_allclose(hml.hml_return, -reverse.hml_return)
        flipped_slope = env['run_fama_macbeth'](flipped, 'signal', 'test', 'test', 'loss')
        np.testing.assert_allclose(slope.signal_slope, -flipped_slope.signal_slope)


if __name__ == '__main__':
    unittest.main()
