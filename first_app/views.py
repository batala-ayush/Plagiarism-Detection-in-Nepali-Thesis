# Create your views here.

from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import render


from docx import Document

#importing the local preprocessing function
from .tfidf_similarity import compute_tfidfsimilarity
from .fingerprint_similarity import compute_fingerprint_similarity
from .word_similarity import compute_word_similarity
from .ngram_similarity import compute_ngram_similarity
from .models import thesis_docx
from .predict import predict_lab
from .forms import DocxUploadForm
from .highlight_paragraphs import highlight_paragraph


def homepage(request):
    return render(request, 'plag_check.html')
    
def thesis_upload_page(request):
    return render(request, 'thesis_upload.html')

def thesis_upload_succesfull_page(request):
    print('agadi')
    print(request)
    if request.method == 'POST':
        print('paxadi')
        print(request.FILES)

        got_thesis_file = request.FILES.get("thesis_file_docx")
        print(got_thesis_file)
        thesis_data = thesis_docx(thesis = got_thesis_file)
        thesis_data.save()
        return HttpResponse("Thesis Uploaded Sucessfully")


def count_total_words(paragraphs):
    total_words = 0
    for paragraph in paragraphs:
        total_words += len(paragraph.split())
    return total_words


def count_total_words_one_paragraph(paragraph):
    total_words = 0
    total_words += len(paragraph.split())
    return total_words



def plagiarism_check(request):
    if request.method == 'POST':
        form = DocxUploadForm(request.POST, request.FILES)
        if form.is_valid():
            docx_file = request.FILES['docx_file']
            docx_file_title_name = docx_file.name
            docx_file_author_name = 'Inputed_Thesis_Author_Name'
            print(type(docx_file))
            sus_document = Document(docx_file)

            print(type(sus_document))
            input_paragraphs = [para.text.strip() for para in sus_document.paragraphs if para.text.strip()]  # Filter out empty or whitespace paragraphs
            # paragraphs = get_paragraphs_from_word_file(docx_file)
            print(input_paragraphs)
            # print(type(paragraphs))
            #return render(request, 'result.html', {'paragraphs': paragraphs})

            docx_files = thesis_docx.objects.all()
            file_names = [docx_file.thesis.name for docx_file in docx_files]
            print(file_names)
            print(file_names[0])
            paragraphs_list = []

            for docx_file in docx_files:
                document = Document(docx_file.thesis)
                #paragraphs = [para.text for para in document.paragraphs]
                para = [para.text.strip() for para in document.paragraphs if para.text.strip()]  # Filter out empty or whitespace paragraphs
                #paragraphs = get_paragraphs_from_word_file(document)
                paragraphs_list.append(para)


            plagiarised_paragraphs =[]
            # Loop through both lists simultaneously and call the function
            for para1 in input_paragraphs:
                paragraphs_list_counter =0
                for para2_group in paragraphs_list:
                    max_avg_feature_sim =0
                    max_sim_database_paragraph = ''
                    is_label_one = 0
                    for para2 in para2_group:
                        #asma two paragraph aaucha
                        ngram_sim = overlap_similarity(para1,para2)
                        tfidf_sim = calculate_tfidfsimilarity(para1,para2)
                        finger_sim = fingerprint_similarity(para1,para2)
                        word_sim = compute_wordsimilarity(para1,para2)
                        label = predict_lab(ngram_sim,finger_sim,word_sim,tfidf_sim)
                        avg_feature_sim = (ngram_sim + tfidf_sim + finger_sim + word_sim)/4
                        if(avg_feature_sim > max_avg_feature_sim and label == 1):
                            max_avg_feature_sim = avg_feature_sim
                            max_sim_database_paragraph = para2
                        
                        if(label == 1):
                            is_label_one = 1
                    if is_label_one ==1 :
                        my_dict = {"paragraph": para1 ,"database_paragraph":max_sim_database_paragraph, "source":file_names[paragraphs_list_counter], "average_feature_score": max_avg_feature_sim}
                        plagiarised_paragraphs.append(my_dict)
                        #asma lekhne sentence wala code
                        #
                    is_label_one = 0
                    paragraphs_list_counter = paragraphs_list_counter + 1
                #         print(para1)
                #         print("-----")
                #         print(para2)
                #         print("label: ")
                #         print(label)
                #     print("changeeeeeeeeeeee1111111")
                # print("changeeeeeeeeeeee22222222")
            print(plagiarised_paragraphs)
            #return render(request, 'try.html', {'paragraphs_list': paragraphs_list})
            #for highlighting docx
            #output_file_path = "highlighted_document.docx"  # Replace with the desired output file path
            #target_paragraphs = plagiarised_paragraphs
            #process_plagiarized_paragraphs(sus_document, output_file_path, target_paragraphs)
            final_plagiarised_paragraphs = {}
            for entry in plagiarised_paragraphs:
                paragraph = entry['paragraph']
                if paragraph in final_plagiarised_paragraphs:
                    if entry['average_feature_score'] > final_plagiarised_paragraphs[paragraph]['average_feature_score']:
                        final_plagiarised_paragraphs[paragraph] = entry
                else:
                    final_plagiarised_paragraphs[paragraph] = entry

            #for highlighting docx
            final_plagiarised_paragraphs = list(final_plagiarised_paragraphs.values())
            print(final_plagiarised_paragraphs)
            total_words_input_docx = count_total_words(input_paragraphs)
            print(total_words_input_docx)
            # Calculate and store the similarity index for each paragraph
            similarity_data = []
            for entry in final_plagiarised_paragraphs:
                sim_index = round((((count_total_words_one_paragraph(entry['paragraph']))*entry['average_feature_score'])/ total_words_input_docx)*100,2)
                similarity_data.append({'input_paragraph':entry['paragraph'],'source': entry['source'],'database_paragraph':entry['database_paragraph'], 'sim_index': sim_index})
            print(similarity_data)

            #final wala dictionary aba
            grouped_paragraphs = []
            for entry in similarity_data:
                source = entry['source']
                input_paragraph = entry['input_paragraph']
                database_paragraph = entry['database_paragraph']
                sim_index = entry['sim_index']
                
                # Initialize a flag to False
                contains_source = False
                for single_dict in grouped_paragraphs:
                    if 'source' in single_dict and single_dict['source'] == source:
                        single_dict['input_paragraphs'].append(input_paragraph)
                        single_dict['database_paragraphs'].append(database_paragraph)
                        single_dict['percentage'] += sim_index
                        contains_source = True
                        break
                if(contains_source == False):
                    grouped_paragraphs.append({'input_paragraphs':[entry['input_paragraph']],'source': entry['source'],'database_paragraphs':[entry['database_paragraph']], 'percentage': sim_index, 'author': 'Author_name'})
            print(grouped_paragraphs)

            #highlight wala
            highlight_new_wala(grouped_paragraphs,sus_document,docx_file_title_name,docx_file_author_name)
            
            context = {'result_list': similarity_data}
            return render(request, 'try1.html', context)
    else:
        form = DocxUploadForm()
    return render(request, 'upload.html', {'form': form})

