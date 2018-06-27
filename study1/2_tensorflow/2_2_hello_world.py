'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import tensorflow as tf

# 상수를 생성하여 기본 그래프에 추가 (상수 명령값을 출력하는 텐서를 하나 만든다)
hello = tf.constant('Hello, World!')

# 세션 시작
with tf.Session() as sess:
    # 명령 실행
    print(sess.run(hello))

    print(hello.eval()) # eval()는 with 구문과 같이 사용해야 함
