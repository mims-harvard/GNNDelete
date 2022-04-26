import boto3
import awswrangler as wr
import pandas as pd
from .s3io import read_txt_s3, scipy_loadmat_s3


def get_binding_data_union(stage='development', boto3_session=None, gene_id_mapping_dict=None, food_chem=None):

    if stage == 'default':
        col = ['chemical', 'ncbi']
        cg = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/network_proximity/chemical_protein/binding_data_union.csv',
            usecols=col, 
            boto3_session=boto3_session)
        # cg.chemical = cg.chemical.apply(str.lstrip, args=('CIDms',))
        # cg.protein = cg.protein.apply(str.lstrip, args=('9606.',))

        # Map gene ID   
        # if gene_id_mapping_dict is not None:
        #     cg['ncbi'] = cg.protein.apply(gene_id_mapping_dict.get)
        # else:
        #     cg['ncbi'] = cg.protein
        
        # cg = cg.drop('protein', axis=1)
        cg = cg.dropna()

        cg.chemical = cg.chemical.apply(int)

        # Select only food chemicals
        if food_chem is not None:
            cg = cg[cg.chemical.isin(food_chem)]

        cg.chemical = cg.chemical.apply(str)
        cg.ncbi = cg.ncbi.apply(str)
        # cg = cg[(~cg.chemical.str.contains('[A-Za-z]')) & (~cg.ncbi.str.contains('[A-Za-z]'))]
        cg = cg.drop_duplicates()

    elif stage == 'development':
        query = 'select chemical, protein from hsproteinchemicaldetailed where experimental > 0'
        cg = wr.athena.read_sql_query(query, database='stitch-source-dbs', ctas_approach=False, boto3_session=boto3_session)
        cg.chemical = cg.chemical.apply(str.lstrip, args=('CIDms',))
        cg.protein = cg.protein.apply(str.lstrip, args=('9606.',))

        # Map gene ID
        if gene_id_mapping_dict is not None:
            cg['ncbi'] = cg.protein.apply(gene_id_mapping_dict.get)
        else:
            cg['ncbi'] = cg.protein

        cg = cg.drop('protein', axis=1)
        cg = cg.dropna()

        cg.chemical = cg.chemical.apply(int)

        # Select only food chemicals
        if food_chem is not None:
            cg = cg[cg.chemical.isin(food_chem)]

        cg.chemical = cg.chemical.apply(str)
        cg.ncbi = cg.ncbi.apply(str)
        cg = cg[(~cg.chemical.str.contains('[A-Za-z]')) & (~cg.ncbi.str.contains('[A-Za-z]'))]
        cg = cg.drop_duplicates()
        cg.ncbi = cg.ncbi.apply(float).apply(int).apply(str)

    else:
        raise NotImplementedError
    
    # Add edge type and edge meta type
    cg['edge_type'] = 'binding'
    cg['edge_meta_type'] = 'chemical-protein'

    # Rename columns. Add identifiers for entity types
    cg = cg.rename(columns={'chemical': 'source', 'ncbi': 'target'})
    cg.source = cg.source.apply(lambda x: 'c' + str(x))
    cg.target = cg.target.apply(lambda x: 'g' + str(x))

    return cg


def get_disease_gene_guney(stage='development', boto3_session=None):

    if stage == 'default':
        dg = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/network_proximity/disease_protein/disease_gene_guney.csv', 
            boto3_session=boto3_session)
        dg.ncbi = dg.ncbi.apply(str)
        dg.disease = dg.disease.str.lower()

    elif stage == 'development':
        dg = wr.athena.read_sql_table('diseasegeneguney', database='ppi-dbs', boto3_session=boto3_session)
        dg.ncbi = dg.ncbi.apply(str)
        dg.disease = dg.disease.str.lower()

    else:
        raise NotImplementedError
    
    # Add edge type and edge meta type
    dg['edge_type'] = 'target'
    dg['edge_meta_type'] = 'disease-protein'

    # Rename columns. Add identifiers for entity types
    dg = dg.rename(columns={'disease': 'source', 'ncbi': 'target'})
    dg.source = dg.source.apply(lambda x: 'd' + str(x))
    dg.target = dg.target.apply(lambda x: 'g' + str(x))

    return dg


def get_ppi_2019(stage='development', boto3_session=None):
    
    if stage == 'default':
        col = ['source', 'target']
        ppi = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/network_proximity/ppi/interactome_2019_merged/interactome_2019_merged.csv', 
            usecols=col, 
            boto3_session=boto3_session)

    elif stage == 'development':
        query = 'select proteina, proteinb from ppiinteractome2019merged'
        ppi = wr.athena.read_sql_query(query, database='ppi-dbs', boto3_session=boto3_session)

    else:
        raise NotImplementedError
    
    ppi['edge_type'] = 'protein-protein'
    ppi['edge_meta_type'] = 'protein-protein'

    # Rename columns. Add identifiers for entity types
    ppi = ppi.rename(
        columns={'proteina': 'source', 'proteinb': 'target', 'proteinA': 'source', 'proteinB': 'target'})
    ppi.source = ppi.source.apply(lambda x: 'g' + str(x))
    ppi.target = ppi.target.apply(lambda x: 'g' + str(x))

    return ppi


def get_ppi_all(stage='development', boto3_session=None):
    
    if stage == 'default':
        col = ['source', 'target']
        ppi = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/network_proximity/ppi/all_interactions/all_interactions.csv', 
            usecols=col, 
            boto3_session=boto3_session)

    else:
        raise NotImplementedError
    
    ppi['edge_type'] = 'protein-protein'
    ppi['edge_meta_type'] = 'protein-protein'

    ppi.source = ppi.source.apply(lambda x: 'g' + str(x))
    ppi.target = ppi.target.apply(lambda x: 'g' + str(x))

    return ppi


def get_ppi_customfilt(stage='development', boto3_session=None):
    
    if stage == 'default':
        col = ['source', 'target']
        ppi = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/network_proximity/ppi/customfilt_interactions/customfilt_interactions.csv', 
            usecols=col, 
            boto3_session=boto3_session)

    else:
        raise NotImplementedError
    
    ppi['edge_type'] = 'protein-protein'
    ppi['edge_meta_type'] = 'protein-protein'

    ppi.source = ppi.source.apply(lambda x: 'g' + str(x))
    ppi.target = ppi.target.apply(lambda x: 'g' + str(x))

    return ppi

def get_ppi_evidence2(stage='development', boto3_session=None):
    
    if stage == 'default':
        col = ['source', 'target']
        ppi = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/network_proximity/ppi/evidence2_interactions/evidence2_interactions.csv', 
            usecols=col, 
            boto3_session=boto3_session)

    else:
        raise NotImplementedError
    
    ppi['edge_type'] = 'protein-protein'
    ppi['edge_meta_type'] = 'protein-protein'

    ppi.source = ppi.source.apply(lambda x: 'g' + str(x))
    ppi.target = ppi.target.apply(lambda x: 'g' + str(x))

    return ppi

def get_ppi_inwebfilt(stage='development', boto3_session=None):
    
    if stage == 'default':
        col = ['source', 'target']
        ppi = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/network_proximity/ppi/inweb_filt_interactions/inweb_filt_interactions.csv', 
            usecols=col, 
            boto3_session=boto3_session)

    else:
        raise NotImplementedError
    
    ppi['edge_type'] = 'protein-protein'
    ppi['edge_meta_type'] = 'protein-protein'

    ppi.source = ppi.source.apply(lambda x: 'g' + str(x))
    ppi.target = ppi.target.apply(lambda x: 'g' + str(x))

    return ppi


def get_ctd(stage='development', boto3_session=None):
    
    if stage == 'default':
        col = ['cid', 'disease_name', 'relation']
        cd = wr.s3.read_csv('s3://kg-data-normalized/ctd_inchi.csv', usecols=col, boto3_session=boto3_session)
        cd = cd.dropna(subset=['cid'])
        cd.cid = cd.cid.apply(int).apply(str)
        cd.disease_name = cd.disease_name.str.lower()
        cd = cd[col]

    elif stage == 'development':
        query = 'SELECT chemical_name, disease_name, relation FROM parquet_ctdchemicalsdiseases'
        cd = wr.athena.read_sql_query(query, database='ctd-dbs', boto3_session=boto3_session)
        cd.chemical_name = cd.chemical_name.apply(str.lower)
        mesh2cid = wr.athena.read_sql_table('cid_mesh', 'pubchem-dbs', boto3_session=boto3_session)
        mesh2cid.mesh_term = mesh2cid.mesh_term.apply(str.lower)
        cd = pd.merge(cd, mesh2cid, left_on='chemical_name', right_on='mesh_term')
        cd = cd[['cid', 'disease_name', 'relation']]
        cd.disease_name = cd.disease_name.str.lower()

    else:
        raise NotImplementedError
    
    # Add edge type and edge meta type
    cd['edge_meta_type'] = 'chemical-disease'

    # Rename columns. Add identifiers for entity types
    cd = cd.rename(columns={'cid': 'source', 'disease_name': 'target', 'relation': 'edge_type'})
    cd.source = cd.source.apply(lambda x: 'c' + x)
    cd.target = cd.target.apply(lambda x: 'd' + x)

    return cd


def get_food_chem(stage='development', boto3_session=None):
    
    if stage == 'default':
        food_chem = wr.s3.read_csv(
            's3://kg-data-normalized/NDM_Master_cid.csv', 
            usecols=['PubChem'],
            boto3_session=boto3_session)
        food_chem = food_chem['PubChem'].dropna().apply(int).unique()

        # col = ['ensembl_protein_id', 'gene_ncbi_id']
        # gene = pd.read_csv('../network_proximity/input/node_gene.csv', usecols=col)
        # gene = gene[gene.ensembl_protein_id.str.startswith('ENSP')]
        # gene = {i: j for i, j in zip(gene.ensembl_protein_id, gene.gene_ncbi_id)}
    
    elif stage == 'development':
        food_chem = wr.athena.read_sql_table('ndmmaster', 'ppi-dbs', boto3_session=sess)
        food_chem = food_chem.pubchem.dropna().apply(int).unique()

        # query = 'select protein_stable_id, xref from ensembl2ncbi'
        # gene = wr.athena.read_sql_query(query, database='ensembl-dbs', boto3_session=sess)
        # gene = gene[(gene.protein_stable_id.str.startswith('ENSP')) & (gene.xref.apply(str).str.isdigit())]
        # gene = {i: j for i, j in zip(gene.protein_stable_id, gene.xref)}

    else:
        raise NotImplementedError
    
    return food_chem


def get_chemical_phenotype(stage='default', boto3_session=None, cid_subset=None):
    
    assert stage == 'default', 'Intermediate phenotype data is only available in the default AWS account'
    
    phenotypes = set()
    chem_phenotype = []
    phenotype_bucket = boto3.resource('s3').Bucket(name='intermediate-phenotypes')
    for i in phenotype_bucket.objects.all():
        if i.key.startswith('ground-truth/') and i.key.endswith('_therapeutics.csv'):
            df = wr.s3.read_csv(f's3://intermediate-phenotypes/{i.key}', usecols=['cid'])        
            dname = i.key[len('ground-truth/'):][:-len('_therapeutics.csv')].lower()
            dname = dname.replace('_', ' ')
            df = df.rename(columns={'cid': 'source'})
            df['target'] = dname
            chem_phenotype.append(df)
            phenotypes.add(dname)
    cp = pd.concat(chem_phenotype).drop_duplicates()
    cp.source = cp.source.apply(int)

    if cid_subset is not None:
        cp = cp[cp.source.isin(cid_subset)]
    
    phenotypes = sorted(list(phenotypes))

    # Add edge type and edge meta type
    cp['edge_type'] = 'therapeutic'
    cp['edge_meta_type'] = 'chemical-disease'

    # Rename columns. Add identifiers for entity types
    # cp = cp.rename(columns={'cid': 'source', ''})
    cp.source = cp.source.apply(lambda x: 'c' + str(x))
    cp.target = cp.target.apply(lambda x: 'p' + str(x))

    return cp, phenotypes


def get_phenotype_gene(stage='default', boto3_session=None):

    assert stage == 'default', 'Intermediate phenotype data is only available in the default AWS account'

    pg = wr.s3.read_csv(
        's3://intermediate-phenotypes/input-data/IntermediatePhenotypes_genes.csv', 
        boto3_session=boto3_session)
    pg = pg.dropna()
    pg.ncbi = pg.ncbi.apply(int)
    pg.disease = pg.disease.apply(str.lower).str.replace('/', ' ')
    pg.loc[pg.disease=='hormonal health', 'disease'] = 'hormone health'    # They are the same phenotype
    
    phenotypes = sorted(list(pg.disease.unique()))
    
    # Add edge type and edge meta type
    pg['edge_type'] = 'target'
    pg['edge_meta_type'] = 'disease-gene'
    
    # Rename columns. Add identifiers for entity types
    pg = pg.rename(columns={'disease': 'source', 'ncbi': 'target'})
    pg.source = pg.source.apply(lambda x: 'p' + str(x))
    pg.target = pg.target.apply(lambda x: 'g' + str(x))

    return pg, phenotypes
    

def get_drug_target_curated_original(stage='development', boto3_session=None):

    if stage == 'default':
        cg = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-targets-all.csv', 
            boto3_session=boto3_session)

    elif stage == 'development':
        cg = wr.s3.read_csv(
            's3://foodome-development-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-targets-all.csv', 
            boto3_session=boto3_session)

    else:
        raise NotImplementedError
    
    # Add edge type and edge meta type
    cg['edge_type'] = 'binding'
    cg['edge_meta_type'] = 'chemical-protein'
    
    # Rename columns. Add identifiers for entity types
    cg = cg.rename(columns={'STITCH': 'source', 'Gene': 'target'})
    cg.source = cg.source.apply(lambda x: 'c' + str(x))
    cg.target = cg.target.apply(lambda x: 'g' + str(x))

    return cg


def get_drug_target_all_original(stage='development', boto3_session=None):

    if stage == 'default':
        cg = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-targets.csv')

    elif stage == 'development':
        cg = wr.s3.read_csv(
            's3://foodome-development-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-targets.csv')

    else:
        raise NotImplementedError

    # Add edge type and edge meta type
    cg['edge_type'] = 'binding'
    cg['edge_meta_type'] = 'chemical-protein'
    
    # Rename columns. Add identifiers for entity types
    cg = cg.rename(columns={'STITCH': 'source', 'Gene': 'target'})
    cg.source = cg.source.apply(lambda x: 'c' + str(x))
    cg.target = cg.target.apply(lambda x: 'g' + str(x))

    return cg


def get_ppi_original(stage='development', boto3_session=None):

    if stage == 'default':
        ppi = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-ppi.csv', 
            boto3_session=boto3_session)

    elif stage == 'development':
        ppi = wr.s3.read_csv(
            's3://foodome-development-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-ppi.csv', 
            boto3_session=boto3_session)

    else:
        raise NotImplementedError
    
    # Add edge type and edge meta type
    ppi['edge_type'] = 'protein_protein_interaction'
    ppi['edge_meta_type'] = 'protein-protein'

    # Rename columns. Add identifiers for entity types
    ppi = ppi.rename(
        columns={'Gene 1': 'source', 'Gene 2': 'target'})
    ppi.source = ppi.source.apply(lambda x: 'g' + str(x))
    ppi.target = ppi.target.apply(lambda x: 'g' + str(x))

    return ppi


def get_drug_drug_original(stage='development', boto3_session=None):

    if stage == 'default':
        ddi = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-combo.csv')

    elif stage == 'development':
        ddi = wr.s3.read_csv(
            's3://foodome-development-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-combo.csv')

    else:
        raise NotImplementedError
    
    # Add edge type and edge meta type
    ddi['edge_meta_type'] = 'drug-drug'
    
    # Rename columns. Add identifiers for entity types
    ddi = ddi.rename(columns={'STITCH 1': 'source', 'STITCH 2': 'target', 'Polypharmacy Side Effect': 'edge_type'})
    ddi.source = ddi.source.apply(lambda x: 'c' + x)
    ddi.target = ddi.target.apply(lambda x: 'c' + x)

    return ddi


# Original data toy version
def get_drug_target_toy_version(stage='development', boto3_session=None):

    if stage == 'default':
        cg = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-targets.csv')

    elif stage == 'development':
        cg = wr.s3.read_csv(
            's3://foodome-development-prediction-us-east-1/decagon_original_data_toy_version/raw_data/trimmed-marinka-dgi.csv', 
            index_col=0, 
            boto3_session=boto3_session)

    else:
        raise NotImplementedError

    # Add edge type and edge meta type
    cg['edge_type'] = 'binding'
    cg['edge_meta_type'] = 'chemical-protein'
    
    # Rename columns. Add identifiers for entity types
    cg = cg.rename(columns={'STITCH': 'source', 'Gene': 'target'})
    cg.source = cg.source.apply(lambda x: 'c' + str(x))
    cg.target = cg.target.apply(lambda x: 'g' + str(x))

    return cg


def get_ppi_original_toy_version(stage='development', boto3_session=None):

    if stage == 'default':
        ppi = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-ppi.csv', 
            boto3_session=boto3_session)

    elif stage == 'development':
        ppi = wr.s3.read_csv(
            's3://foodome-development-prediction-us-east-1/decagon_original_data_toy_version/raw_data/trimmed-marinka-ppi.csv', 
            index_col=0, 
            boto3_session=boto3_session)

    else:
        raise NotImplementedError
    
    # Add edge type and edge meta type
    ppi['edge_type'] = 'protein_protein_interaction'
    ppi['edge_meta_type'] = 'protein-protein'

    # Rename columns. Add identifiers for entity types
    ppi = ppi.rename(
        columns={'Gene 1': 'source', 'Gene 2': 'target'})
    ppi.source = ppi.source.apply(lambda x: 'g' + str(x))
    ppi.target = ppi.target.apply(lambda x: 'g' + str(x))

    return ppi


def get_drug_drug_original_toy_version(stage='development', boto3_session=None):

    if stage == 'default':
        ddi = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/decagon_original_data/raw_data/bio-decagon-combo.csv')

    elif stage == 'development':
        ddi = wr.s3.read_csv(
            's3://foodome-development-prediction-us-east-1/decagon_original_data_toy_version/raw_data/trimmed-toy-marinka-ddi.csv', 
            index_col=0, 
            boto3_session=boto3_session)

    else:
        raise NotImplementedError
    
    ddi['toy side effects'] = ddi['toy side effects'].apply(lambda x: 'side_effect_' + str(x))
    
    # Add edge type and edge meta type
    ddi['edge_meta_type'] = 'drug-drug'
    
    # Rename columns. Add identifiers for entity types
    ddi = ddi.rename(columns={'STITCH 1': 'source', 'STITCH 2': 'target', 'toy side effects': 'edge_type'})
    ddi.source = ddi.source.apply(lambda x: 'c' + x)
    ddi.target = ddi.target.apply(lambda x: 'c' + x)

    return ddi

def get_ppi_guney2016(stage='default', boto3_session=None):

    if stage == 'default':
        cols = ['gene_1', 'gene_2']
        ppi = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/guney_2016/interactome.tsv',
            sep='\t',
            usecols=cols)

    else:
        raise NotImplementedError
    
    # Add edge type and edge meta type
    ppi['edge_type'] = 'interaction'
    ppi['edge_meta_type'] = 'protein-protein'
    
    # Rename columns. Add identifiers for entity types
    ppi = ppi.rename(columns={'gene_1': 'source', 'gene_2': 'target'})
    ppi.source = ppi.source.apply(lambda x: 'g' + str(x))
    ppi.target = ppi.target.apply(lambda x: 'g' + str(x))

    return ppi

def get_drug_gene_guney2016(stage='default', boto3_session=None):

    if stage == 'default':
        lines = read_txt_s3(
            'guney_2016/drug_gene.csv', 'foodome-default-prediction-us-east-1', boto3.client('s3'))

    else:
        raise NotImplementedError
    
    dg = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        
        r = line.strip().split(',')
        gene = []
        for g in r[1:]:
            if g != '':
                dg.append({
                    'drug': r[0],
                    'gene': g
                })
    dg = pd.DataFrame(dg)

    # Add edge type and edge meta type
    dg['edge_type'] = 'binding'
    dg['edge_meta_type'] = 'chemical-gene'
    
    # Rename columns. Add identifiers for entity types
    dg = dg.rename(columns={'drug': 'source', 'gene': 'target'})
    dg.source = dg.source.apply(lambda x: 'c' + x)
    dg.target = dg.target.apply(lambda x: 'g' + x)

    return dg

def get_disease_gene_guney2016(stage='default', boto3_session=None):

    if stage == 'default':
        cols = ['disease', 'OMIM_genes', 'GWAS_genes']
        dg = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/guney_2016/disease_gene.tsv',
            sep='\t',
            usecols=cols)

    else:
        raise NotImplementedError
    
    gene = pd.concat([
        dg['OMIM_genes'].str.split(';').explode(),
        dg['GWAS_genes'].str.split(';').explode()
    ]).dropna().to_frame('gene')

    dg = dg['disease'].to_frame().join(gene)

    # Add edge type and edge meta type
    dg['edge_type'] = 'target'
    dg['edge_meta_type'] = 'disease-gene'
    
    # Rename columns. Add identifiers for entity types
    dg = dg.rename(columns={'disease': 'source', 'gene': 'target'})
    dg.source = dg.source.apply(lambda x: 'g' + str(x))
    dg.target = dg.target.apply(lambda x: 'g' + str(x))

    return dg

def get_drug_disease_guney2016(stage='default', boto3_session=None):

    if stage == 'default':
        cols = ['Drugbank ID', 'Disease']
        dd = wr.s3.read_csv(
            's3://foodome-default-prediction-us-east-1/guney_2016/drug_disease.csv',
            usecols=cols,
            encoding = 'unicode_escape')

    else:
        raise NotImplementedError
    
    dd = dd.dropna()

    # Add edge type and edge meta type
    dd['edge_type'] = 'therapeutic'
    dd['edge_meta_type'] = 'chemical-disease'
    
    # Rename columns. Add identifiers for entity types
    dd = dd.rename(columns={'Drugbank ID': 'source', 'Disease': 'target'})
    dd.source = dd.source.apply(lambda x: 'c' + str(x))
    dd.target = dd.target.apply(lambda x: 'd' + str(x))

    return dd

def get_ppi_node2vec(stage='default', boto3_session=None):

    import scipy.sparse
    if stage == 'default':
        mat = scipy_loadmat_s3(
            'decagon_node2vec/raw_data/Homo_sapiens.mat', 'foodome-default-prediction-us-east-1', boto3.client('s3'))
        network = scipy.sparse.coo_matrix(mat['network'])
        ppi = []
        for r, c in zip(network.row, network.col):
            ppi.append({
                'source': f'node{r}',
                'target': f'node{c}'
            })
        ppi = pd.DataFrame(ppi)

    else:
        raise NotImplementedError
    
    # Add edge type and edge meta type
    ppi['edge_type'] = 'interaction'
    ppi['edge_meta_type'] = 'protein-protein'
    
    # Rename columns. Add identifiers for entity types
    ppi = ppi.rename(columns={'gene_1': 'source', 'gene_2': 'target'})
    ppi.source = ppi.source.apply(lambda x: 'g' + str(x))
    ppi.target = ppi.target.apply(lambda x: 'g' + str(x))

    return ppi

def get_label_node2vec(stage='default', boto3_session=None):

    import scipy.sparse
    if stage == 'default':
        mat = scipy_loadmat_s3(
            'decagon_node2vec/raw_data/Homo_sapiens.mat', 'foodome-default-prediction-us-east-1', boto3.client('s3'))
        group = scipy.sparse.coo_matrix(mat['group'])
        label = []
        for r, c in zip(group.row, group.col):
            label.append({
                'source': f'node{r}',
                'target': f'label{c}'
            })
        label = pd.DataFrame(label)

    else:
        raise NotImplementedError
    
    # Add edge type and edge meta type
    label['edge_type'] = 'label'
    label['edge_meta_type'] = 'protein-label'
    
    # Rename columns. Add identifiers for entity types
    label = label.rename(columns={'gene_1': 'source', 'gene_2': 'target'})
    label.source = label.source.apply(lambda x: 'g' + str(x))
    label.target = label.target.apply(lambda x: 'l' + str(x))

    return label
