import os
import numpy as np
import sys

def compute_TF(file_paths,file_type,method = 'normal',with_DF = False,document_file_names = None):
    TF_list = []
    DF = {}
    for index,path in enumerate(file_paths):
        TF = {}
        file_name = document_file_names[index] if file_type == 'Document' else None
        
        with open(path,'r') as file:
            indexterms = file.read().split()

        for term in indexterms:
            TF[term] = 1 if term not in TF else TF[term] + 1
            if with_DF:
                if term in DF:
                    if file_name not in DF[term]:
                        DF[term].append(file_name)
                else:
                    DF[term] = [file_name]

        TF_list.append(compute_TF_methoded(TF,method))
        print(f"finish computing:{index}",end = '\r')
    if with_DF:
        return TF_list,DF
    else:
        return TF_list

def compute_TF_methoded(TF,method):
    tempTF = {}
    
    for term,value in TF.items():
        if method == 'normal':
            tempTF[term] = value
        elif method == 'DoubleNormalization.5':
            tempTF[term] = (.5 + .5 * value / max(TF.values()))
        else:
            print('method error!!!')
            sys.exit()
    return tempTF

def compute_IDF(DF,document_num,method = 'normal'):
    IDF = {}
    for term,value in DF.items():
        if method == 'normal':
            IDF[term] = np.log10(document_num / len(value))**2
        elif method == 'Smooth':
            IDF[term] = np.log2(1 + (document_num / DF))
        else:
            print('method error!!!')
    return IDF

def compute_TFIDF(TF_list,IDF):
    TFIDF_list = []
    for TF in TF_list:
        TFIDF = {}
        for term in TF:
            TFIDF[term] = TF[term] * IDF[term] if term in IDF else 0
        TFIDF_list.append(TFIDF)
    return TFIDF_list

def main():
    #current path:C:\Users\88698\Desktop\NTUST\IR\Week1
    #file path:C:\Users\c88698\Desktop\NTUST\IR\Week1\ntust-ir-2020
    #Get query file path
    queries_file_names = [path_imfor[2] for path_imfor in os.walk('ntust-ir-2020\queries')][0]
    queries_root_path = [path_imfor[0] for path_imfor in os.walk('ntust-ir-2020\queries')][0]
    queries_file_paths = [os.path.join(queries_root_path,names) for names in queries_file_names]

    #Get document file path
    document_file_names = [path_imfor[2] for path_imfor in os.walk('ntust-ir-2020\docs')][0]
    document_root_path = [path_imfor[0] for path_imfor in os.walk('ntust-ir-2020\docs')][0]
    document_file_paths = [os.path.join(document_root_path,names) for names in document_file_names]

    #Compute TF,DF
    queryTF_list = compute_TF(queries_file_paths,'Query',method = 'DoubleNormalization.5')
    documentTF_list,DF = compute_TF(document_file_paths,'Document',method = 'normal',with_DF = True,document_file_names = document_file_names)
    document_num = len(documentTF_list)

    #Compute IDF
    IDF = compute_IDF(DF,document_num)

    #Compute TFIDF
    queryTFIDF_list = compute_TFIDF(queryTF_list,IDF)
    documentTFIDF_list = compute_TFIDF(documentTF_list,IDF)

    #Output Answer
    ans_file_name = 'vsm_result5.txt'
    File = open(ans_file_name,'w')
    File.write('Query,RetrievedDocuments')
    for query_index,queryTFIDF in enumerate(queryTFIDF_list):#for each query file
        score = {}
        for doc_index,documentTFIDF in enumerate(documentTFIDF_list):#for each document file
            document_file_name = document_file_names[doc_index][:-4]
            document_vector = []
            query_vector = []
            valcaborary = set(list(documentTFIDF.keys()) + list(queryTFIDF.keys()))
            for term in valcaborary:
                if term in queryTFIDF:
                    query_vector.append(queryTFIDF[term])
                else:
                    query_vector.append(0)
                if term in documentTFIDF:
                    document_vector.append(documentTFIDF[term])
                else:
                    document_vector.append(0)

            length_Q = np.sqrt(sum(value**2 for value in query_vector))
            length_D = np.sqrt(sum(value**2 for value in document_vector))
            if sum(document_vector) == 0 or sum(query_vector) == 0:
                similarity = 0
            else:
                similarity = np.dot(query_vector,document_vector) / (length_Q * length_D)
            score[document_file_name] = similarity
        score = sorted(score.items(), key=lambda x:x[1])
        score.reverse()
        
        query_file_name = queries_file_names[query_index][:-4]
        File.writelines('\n' + query_file_name + ',')
        for (DocumentName,score) in score:
            File.write(DocumentName + ' ')
        print(f"finish processing query:{query_index+1}",end = '\r')

if __name__ == '__main__':
    main()