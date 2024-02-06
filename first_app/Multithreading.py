"""
def find_plagiarised_paragraphs(input_paragraphs):
            # Loop through both lists simultaneously and call the function
            plagiarised_paragraphs =[]
            for para1 in input_paragraphs:
            
                #paragraphs_list_counter =0
                for para2_group in paragraphs_list:
                    max_avg_feature_sim =0
                    max_sim_database_paragraph = ''
                    is_label_one = 0
                    for para2 in para2_group['paragraphs']:
                        #asma two paragraph aaucha
                        #print('--------------------------------------------------------------')
                        #print(para2[0])
                        #print("----------------------------------------------------")
                        ngram_sim = compute_ngram_similarity(para1[0],para2[0])
                        #print(para1[3])
                        #print(para2[3])
                        #print(para1[1])
                        #print(para2[1])
                        finger_sim = compute_fingerprint_similarity(para1[1],para2[1])
                        
                        word_sim = compute_word_similarity(para1[2],para2[2])
                        tfidf_sim = compute_tfidfsimilarity(para1[3],para2[3]) #4th item in list is paragraph 
                        label = predict_lab(ngram_sim,finger_sim,word_sim,tfidf_sim)
                        avg_feature_sim = (ngram_sim + tfidf_sim + finger_sim + word_sim)/4
                        if(avg_feature_sim > max_avg_feature_sim and label == 1):
                            max_avg_feature_sim = avg_feature_sim
                            max_sim_database_paragraph = para2[3]
                        
                        if(label == 1):
                            is_label_one = 1
                    if is_label_one ==1 :
                        my_dict = {"paragraph": para1[3] ,"database_paragraph":max_sim_database_paragraph, "source":para2_group['source'],"average_feature_score": max_avg_feature_sim}
                        plagiarised_paragraphs.append(my_dict)
                        #asma lekhne sentence wala code
                        #
                    is_label_one = 0
                    #paragraphs_list_counter = paragraphs_list_counter + 1
                #         print(para1)
                #         print("-----")
                #         print(para2)
                #         print("label: ")
                #         print(label)
                #     print("changeeeeeeeeeeee1111111")
                # print("changeeeeeeeeeeee22222222")
            #print(plagiarised_paragraphs)
            #return render(request, 'try.html', {'paragraphs_list': paragraphs_list})
            #for highlighting docx
            #output_file_path = "highlighted_document.docx"  # Replace with the desired output file path
            #target_paragraphs = plagiarised_paragraphs
            #process_plagiarized_paragraphs(sus_document, output_file_path, target_paragraphs)
            return plagiarised_paragraphs
        
        no_of_threads = 2
        half = len(input_paragraphs)//no_of_threads
        input_paragraphs_first_half,input_paragraphs_second_half = input_paragraphs[:half],input_paragraphs[half:]


        def worker1(input_paragraphs):
            global plagiarised_paragraphs_first_half

            plagiarised_paragraphs_first_half = find_plagiarised_paragraphs(input_paragraphs)
           

        def worker2(input_paragraphs):
            global plagiarised_paragraphs_second_half

            plagiarised_paragraphs_second_half = find_plagiarised_paragraphs(input_paragraphs)
            
        thread1 = threading.Thread(target=worker1, args=(input_paragraphs_first_half,))
        thread2 = threading.Thread(target=worker2, args=(input_paragraphs_second_half,))

        # Start the threads
        thread1.start()
        thread2.start()

        # Wait for both threads to finish
        thread1.join()
        thread2.join()
        
        total_plagiarised_paragraphs = plagiarised_paragraphs_first_half+plagiarised_paragraphs_second_half







"""