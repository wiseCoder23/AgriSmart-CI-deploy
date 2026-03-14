from flask import Flask, request, jsonify, render_template
import pickle, json, numpy as np, pandas as pd, os
from datetime import datetime

app = Flask(__name__, template_folder='templates')
app.config['JSON_ENSURE_ASCII'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, '..', 'models', 'agrismart_model')

def charger(nom):
    with open(os.path.join(MODEL_DIR, f'{nom}.pkl'), 'rb') as f:
        return pickle.load(f)

ridge           = charger('ridge')
rf              = charger('rf')
scaler          = charger('scaler')
feat_cols       = charger('feat_cols')
feat_rf         = charger('feat_rf')
utility_matrix  = charger('utility_matrix')
sim_df          = charger('sim_df')
df_ref          = charger('df_reference')
risque_map      = charger('risque_map')
pluvio_opt      = charger('pluvio_opt')
with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
    metadata = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','version':metadata['version'],
        'ridge_r2':metadata['ridge_r2'],'rf_r2':metadata['rf_r2'],
        'timestamp':datetime.now().isoformat()})

@app.route('/cultures', methods=['GET'])
def cultures():
    s = df_ref.groupby('culture')['score_rentabilite'].agg(['mean','count']).reset_index()
    return jsonify({'cultures': s.round(2).to_dict(orient='records')})

@app.route('/profile/<agri_id>', methods=['GET'])
def profile(agri_id):
    d = df_ref[df_ref['agriculteur_id']==agri_id]
    if d.empty: return jsonify({'erreur':f'{agri_id} non trouve'}), 404
    n = int(utility_matrix.loc[agri_id].notna().sum()) if agri_id in utility_matrix.index else 0
    return jsonify({'agriculteur_id':agri_id,'region':d['region'].iloc[0],
        'experience_ans':round(float(d['experience_ans'].mean()),1),
        'capital_fcfa':int(d['capital_initial_fcfa'].mean()),
        'n_notations':n,
        'statut':'Riche' if n>=3 else 'Partiel' if n>=1 else 'Cold Start'})

@app.route('/recommend', methods=['POST'])
def recommend():
    body = request.get_json()
    if not body or 'agriculteur_id' not in body:
        return jsonify({'erreur':'agriculteur_id requis'}), 400
    agri_id = body['agriculteur_id']
    n_reco  = int(body.get('n_recommandations', 5))
    if agri_id not in utility_matrix.index:
        return jsonify({'erreur':f'{agri_id} non trouve'}), 404
    n = int(utility_matrix.loc[agri_id].notna().sum())
    if n>=3:   w_knn,w_c,w_rf,mode = 0.40,0.20,0.40,'hybride_complet'
    elif n>=1: w_knn,w_c,w_rf,mode = 0.20,0.40,0.40,'hybride_partiel'
    else:      w_knn,w_c,w_rf,mode = 0.00,0.50,0.50,'cold_start'
    sims = sim_df[agri_id].drop(agri_id).sort_values(ascending=False).head(3)
    agri = df_ref[df_ref['agriculteur_id']==agri_id].mean(numeric_only=True)
    scores = {}
    for culture in utility_matrix.columns:
        moy_a = float(utility_matrix.loc[agri_id].mean()); num,den = 0,0
        for v,s in sims.items():
            if pd.notna(utility_matrix.loc[v,culture]):
                num += s*(float(utility_matrix.loc[v,culture])-float(utility_matrix.loc[v].mean()))
                den += abs(s)
        sk = moy_a+num/den if den>0 else moy_a
        cout = df_ref[df_ref['culture']==culture]['cout_production_fcfa_ha'].mean()
        cap_ok  = min(1.0,agri['capital_initial_fcfa']/max(cout*agri['superficie_ha'],1))
        clim_ok = max(0,1-abs(agri['pluviometrie_mm']-pluvio_opt.get(culture,1000))/1000)
        exp_ok  = min(1.0,max(0,agri['experience_ans']/20-risque_map.get(culture,0.35)+0.5))
        sc      = (0.35*cap_ok+0.30*clim_ok+0.25*exp_ok+0.10*agri['acces_credit'])*100
        row_rf  = {col:0 for col in feat_rf}
        for fld in ['experience_ans','capital_initial_fcfa','superficie_ha',
                    'pluviometrie_mm','temperature_moy_c','acces_marche_km',
                    'nb_formations_recues','membre_cooperative','acces_credit']:
            if fld in row_rf: row_rf[fld]=float(agri.get(fld,0))
        cd = df_ref[df_ref['culture']==culture].mean(numeric_only=True)
        for fld in ['prix_vente_fcfa_kg','cout_production_fcfa_ha','rendement_kg_ha']:
            if fld in row_rf: row_rf[fld]=float(cd.get(fld,0))
        reg_col  = f"region_{df_ref[df_ref['agriculteur_id']==agri_id]['region'].iloc[0]}"
        cult_col = f'culture_{culture}'
        if reg_col  in row_rf: row_rf[reg_col]=1
        if cult_col in row_rf: row_rf[cult_col]=1
        sr = float(rf.predict(np.array([[row_rf[c] for c in feat_rf]]))[0])
        scores[culture] = w_knn*sk+w_c*sc+w_rf*sr
    top = sorted(scores.items(),key=lambda x:x[1],reverse=True)[:n_reco]
    return jsonify({'agriculteur_id':agri_id,'mode':mode,'n_notations':n,
        'poids':{'knn':w_knn,'contenu':w_c,'rf':w_rf},
        'recommandations':[{'rang':i+1,'culture':c,'score':round(s,2)} for i,(c,s) in enumerate(top)],
        'timestamp':datetime.now().isoformat()})

@app.route('/cold-start', methods=['POST'])
def cold_start():
    body=request.get_json(); region=body.get('region','Centre'); n_reco=int(body.get('n_recommandations',5))
    pop=(df_ref[df_ref['region']==region].groupby('culture')['score_rentabilite']
         .agg(['mean','count']).reset_index())
    pop['score_pop']=pop['mean']*np.log1p(pop['count'])
    pop=pop.sort_values('score_pop',ascending=False).head(n_reco)
    return jsonify({'mode':'cold_start','region':region,
        'recommandations':[{'rang':i+1,'culture':r['culture'],'score_moyen':round(r['mean'],2)}
                            for i,(_,r) in enumerate(pop.iterrows())]})

@app.route('/explain/<agri_id>', methods=['GET'])
def explain(agri_id):
    culture=request.args.get('culture','Manioc')
    d=df_ref[df_ref['agriculteur_id']==agri_id]
    if d.empty: return jsonify({'erreur':f'{agri_id} non trouve'}),404
    agri=d.mean(numeric_only=True)
    cout=df_ref[df_ref['culture']==culture]['cout_production_fcfa_ha'].mean()
    rdt =df_ref[df_ref['culture']==culture]['rendement_kg_ha'].mean()
    prix=df_ref[df_ref['culture']==culture]['prix_vente_fcfa_kg'].mean()
    return jsonify({'agriculteur_id':agri_id,'culture':culture,
        'capital_suffisant':bool(agri['capital_initial_fcfa']>=cout*agri['superficie_ha']),
        'adequation_climatique':bool(abs(agri['pluviometrie_mm']-pluvio_opt.get(culture,1000))<400),
        'roi_estime':round(float((prix*rdt-cout)/max(cout,1)),4),
        'risque':risque_map.get(culture,0.35)})

@app.route('/agriculteur/<agri_id>/cultures', methods=['GET'])
def cultures_par_agriculteur(agri_id):
    if agri_id not in utility_matrix.index:
        return jsonify({'erreur': f'{agri_id} non trouve'}), 404
    testees = utility_matrix.loc[agri_id].dropna().index.tolist()
    if not testees:
        testees = sorted(df_ref['culture'].unique().tolist())
    return jsonify({'agriculteur_id': agri_id, 'cultures': sorted(testees)})

@app.route('/agriculteurs', methods=['GET'])
def agriculteurs():
    ids = sorted(df_ref['agriculteur_id'].unique().tolist())
    return jsonify({'agriculteurs': ids})

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({'score_moyen':round(float(df_ref['score_rentabilite'].mean()),2),
        'score_median':round(float(df_ref['score_rentabilite'].median()),2),
        'meilleure_culture':str(df_ref.groupby('culture')['score_rentabilite'].mean().idxmax()),
        'meilleure_region':str(df_ref.groupby('region')['score_rentabilite'].mean().idxmax()),
        'cultures':sorted(df_ref['culture'].unique().tolist()),
        'regions':sorted(df_ref['region'].unique().tolist())})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
