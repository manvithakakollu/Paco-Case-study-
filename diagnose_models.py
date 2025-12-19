import joblib
import os
import pandas as pd

MODELS = {
    'Win Probability': ('models/win_model.joblib', 'data/win_probability_data.csv'),
    'Project Cost': ('models/project_cost_model.joblib', 'data/project_cost_overrun_data.csv'),
    'Time-to-Fill': ('models/time_to_fill_model.joblib', 'data/time_to_fill_data.csv'),
    'Exit Risk': ('models/exit_risk_model.joblib', 'data/employee_exit_risk_data.csv'),
}


def summarize_model(path):
    if not os.path.exists(path):
        return {'exists': False}
    m = joblib.load(path)
    info = {'exists': True, 'type': type(m).__name__}
    # If Prophet fallback saved a dict
    if isinstance(m, dict) and 'forecast' in m:
        info['prophet_forecast'] = True
        info['forecast_rows'] = len(m.get('forecast'))
        return info
    pre = None
    if hasattr(m, 'named_steps'):
        pre = m.named_steps.get('pre')
    if pre is None:
        info['preprocessor'] = None
        return info
    transformers = getattr(pre, 'transformers', [])
    cols = []
    details = []
    for t in transformers:
        # t may be (name, transformer, columns)
        if len(t) >= 3:
            name, transformer, columns = t[0], t[1], t[2]
            # normalize columns
            try:
                if isinstance(columns, (list, tuple)):
                    col_list = list(columns)
                else:
                    col_list = list(columns)
            except Exception:
                col_list = [columns]
            cols.extend([str(c) for c in col_list])
            details.append({'name': name if isinstance(name, str) else str(name), 'n_cols': len(col_list)})
    info['preprocessor'] = {'columns': cols, 'details': details}
    return info


if __name__ == '__main__':
    for model_name, (m_path, data_path) in MODELS.items():
        print('---')
        print('Model:', model_name)
        print('Model file:', m_path, 'exists=', os.path.exists(m_path))
        info = summarize_model(m_path)
        print('Model type:', info.get('type'))
        if info.get('prophet_forecast'):
            print('Prophet forecast saved (rows):', info.get('forecast_rows'))
        pre = info.get('preprocessor')
        if pre:
            print('Preprocessor expects columns (sample):', pre['columns'][:50])
            print('Preprocessor details:', pre['details'])
        else:
            print('No preprocessor info available')

        print('Data file:', data_path, 'exists=', os.path.exists(data_path))
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            print('Data columns:', list(df.columns))
            if pre and pre.get('columns'):
                expected = set(pre['columns'])
                present = set(df.columns)
                missing = expected - present
                extra = present - expected
                print('Missing columns (expected by pipeline):', list(missing)[:50])
                print('Extra columns in data:', list(extra)[:50])
        else:
            print('Data file not found; cannot compare columns')
        print()