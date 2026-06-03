class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int m = text1.size();
        int n = text2.size();

        int T[m+1][n+1];

        for(int i = 0 ; i <= m ; i++){
            for(int j = 0 ; j <= n ; j++){
                if( i == 0 || j == 0)
                    T[i][j] = 0;
            }
        }

         for(int i = 1 ; i <= m ; i++){
            for(int j = 1 ; j <= n ; j++){
                if( text1[i-1] == text2[j-1]){
                    T[i][j] = 1 + T[i-1][j-1];
                }
                else{
                    T[i][j] = max( T[i-1][j] , T[i][j-1]);
                }
            }
        }
        return T[m][n];
    }
};
