{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e1ff28b8",
   "metadata": {},
   "source": [
    "# SMS SPAM Classification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b2082df5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding\n",
    "from tensorflow.keras.optimizers import RMSprop\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing import sequence\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from tensorflow.keras.callbacks import EarlyStopping\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "edbcdfa0",
   "metadata": {},
   "source": [
    "# Load the data into Pandas dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "fcb72555",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>v1</th>\n",
       "      <th>v2</th>\n",
       "      <th>Unnamed: 2</th>\n",
       "      <th>Unnamed: 3</th>\n",
       "      <th>Unnamed: 4</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>ham</td>\n",
       "      <td>Go until jurong point, crazy.. Available only ...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>ham</td>\n",
       "      <td>Ok lar... Joking wif u oni...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>spam</td>\n",
       "      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>ham</td>\n",
       "      <td>U dun say so early hor... U c already then say...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>ham</td>\n",
       "      <td>Nah I don't think he goes to usf, he lives aro...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     v1                                                 v2 Unnamed: 2  \\\n",
       "0   ham  Go until jurong point, crazy.. Available only ...        NaN   \n",
       "1   ham                      Ok lar... Joking wif u oni...        NaN   \n",
       "2  spam  Free entry in 2 a wkly comp to win FA Cup fina...        NaN   \n",
       "3   ham  U dun say so early hor... U c already then say...        NaN   \n",
       "4   ham  Nah I don't think he goes to usf, he lives aro...        NaN   \n",
       "\n",
       "  Unnamed: 3 Unnamed: 4  \n",
       "0        NaN        NaN  \n",
       "1        NaN        NaN  \n",
       "2        NaN        NaN  \n",
       "3        NaN        NaN  \n",
       "4        NaN        NaN  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(r'spam.csv',encoding='latin-1')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "1c25ea5a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 5572 entries, 0 to 5571\n",
      "Data columns (total 2 columns):\n",
      " #   Column  Non-Null Count  Dtype \n",
      "---  ------  --------------  ----- \n",
      " 0   v1      5572 non-null   object\n",
      " 1   v2      5572 non-null   object\n",
      "dtypes: object(2)\n",
      "memory usage: 87.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "79f46bd0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "D:\\users\\meyyappan\\Anaconda\\lib\\site-packages\\seaborn\\_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Number of ham and spam messages')"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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mujVV1/ViRMnNGrUKPXq1UsRERFyd3fX/PnzdeDAAY0ePVqPPfaYPvvsM/O9CQkJGjdunB555BGNGzdOeXl5mj59utq2batNmzapUaNGkqSePXvq3LlzmjZtmm688Ub9+uuv2rBhg7nfCq9/a9u2rebPn6+qVavqf//7n5KTk5WXlydvb2+zbujQobrxxhslXQiCw4cP1//+9z+NHz/e7OuRRx7RBx98oPj4eHXs2FG7du1S3759lZ2d7bTtmZmZatWqlSpUqKDx48erXr162rhxo6ZMmaL9+/frnXfekSQtWbJEsbGxGj58uF588UVVqFBBP/74o3bt2nXZ/YsywADKgXfeeceQZGzevNnIzc016tata7Ro0cI4f/68YRiG0b59e+OWW24x6/ft22dIMt55550iy5JkTJgwwZyeMGGCIcmYMWOGU92tt95qSDI+/vhjcyw/P98ICAgw7rnnHnNszZo1hiSjWbNmZj+GYRj79+833N3djccee8wca9CggXHbbbcZ+fn5TuuKiooyatSoYZw7d85pex9++OEr2j9PP/20IclIS0tzGn/88ccNm81m7N271xyrVauWcdddd13RciUZQUFBRnZ2tjmWmZlpVKhQwUhMTPzD9xUUFBh5eXlGWFiY8dRTT5njhfuqV69eTvVxcXGGJGPEiBFO43369DH8/PyuqNdC58+fN/Lz840DBw4YkoxPP/3UnFf4WU+bNs3pPbGxsUalSpXMz++LL74wJBkvv/yyU93zzz9f5OenOFFRUcatt956yZrCXn6/fwzDMBYtWmRIMt57771Lbt+6desMScZ3331nzhswYIAhyfjoo4/MscKfWUnGt99+a44fP37cqFixojFy5MhL9ln4u9S0aVPz59MwDGPWrFmGJKN3795O9YWfpcPhMAzDMH7++WfDzc3NGD58uFPdyZMnjeDgYKNfv36GYRjGr7/+akgyZs2a9Ye9fPjhh4YkIyMj45I9/965c+eM/Px8Y/LkyYa/v7/5Ge/cudOQZIwZM8apfvHixYYkY8CAAebY0KFDjcqVKxsHDhxwqn3xxRcNScbOnTsNwzCMJ5980qhateoV94ayhVNmKHc8PDw0ZcoUbdmypcippr8iKirKabphw4ay2WxORwXc3Nx00003FXunW3R0tGw2mzldq1YttWnTRmvWrJF04bTSnj179OCDD0qSCgoKzFfPnj11+PBh7d2712mZ99577xX1vnr1ajVq1EitWrVyGh84cKAMwyhyFOHPuPPOO1WlShVzOigoSIGBgU77oKCgQAkJCWrUqJE8PDzk5uYmDw8P/fDDD9q9e3eRZRa3ryXprrvuKjL+22+/Xfa02dGjRzVs2DCFhobKzc1N7u7u5rVRxa2/d+/eTtNNmjTR2bNnzVM9hZ9Z4WdVKDo6+pJ9FGrVqpW+++47xcbG6ssvvyxyxOH3Ll5Hv3795ObmZvYgXTiVFB0dreDgYFWsWFHu7u5q3759sdtns9nUs2dPc7rwZ7ZGjRq67bbbzHE/P78in+Ol9OzZUxUq/N+fjEt9ZtKFU7iS9OWXX6qgoEAPP/yw0898pUqV1L59e61du9bsp169epo+fbpmzpyprVu3mqd7C916663y8PDQkCFDtGDBAv33v/8tttfVq1erc+fOstvt5v4aP368jh8/bn7G69atk3Rhf//e3/72N7m5OZ88WbFihe68806FhIQ4bUPhvw2Fy2rVqpVOnDih/v3769NPPy1yyhZlG4EI5dIDDzygZs2a6dlnn1V+fv5VWaafn5/TtIeHh7y9vVWpUqUi42fPni3y/uDg4GLHjh8/Lkk6cuSIJGn06NFyd3d3esXGxkpSkX9Ar/QOsOPHjxdbGxISYs4vKX9//yJjnp6eysnJMadHjhyp5557Tn369NHy5cuVlpamzZs3q2nTpk51hYrb15caL25/Fzp//ry6du2qjz/+WPHx8fr666+1adMmpaamSlKx6794mzw9PZ1qjx8/Ljc3tyJ1xX3GxRk7dqxefPFFpaamqkePHvL391enTp20ZcuWIrUXL7NwvYWf2alTp9S2bVulpaVpypQpWrt2rTZv3qyPP/642O37o5/Zi/dt4fil9u3vlfQzK/y5b9myZZGf+w8++MD8mbfZbPr666/VrVs3TZs2Tc2aNVNAQIBGjBihkydPSpLq1aunr776SoGBgXriiSdUr1491atXz+narE2bNqlr166SpHnz5mn9+vXavHmznn32Waf9Vbh/g4KCnPov7nM/cuSIli9fXqT/W265RdL//d7GxMSYpw7vvfdeBQYGKiIiQikpKVe0j+FaXEOEcslms2nq1Knq0qWL3nzzzSLzC/8gXHwR8l8JBpeTmZlZ7FjhP67Vq1eXdOGP5cXXHxW6+Fqb3x9xuhR/f38dPny4yPihQ4ec1n2tvPfee3r44YeVkJDgNP7rr7+qatWq13TdO3bs0HfffaekpCQNGDDAHP/xxx9LvEx/f38VFBTo+PHjTn8ci/uMi+Pm5qaRI0dq5MiROnHihL766is988wz6tatmw4ePGhe61K4zBtuuMGcvni9q1ev1qFDh7R27VrzqJCkYq9HKosKf/Y+/PDDy97RWKtWLb399tuSpO+//15Lly7VxIkTlZeXpzfeeEOS1LZtW7Vt21bnzp3Tli1bNHv2bMXFxSkoKEgPPPCAlixZInd3d61YscIpGH7yySdO6yrcv0eOHCl2/1+8DU2aNNHzzz9fbN+F/+MhXbgu6ZFHHtHp06f173//WxMmTFBUVJS+//577ugs4zhChHKrc+fO6tKliyZPnlzklEpQUJAqVaqkbdu2OY1/+umn16yfxYsXO93FcuDAAW3YsEEdOnSQdCHshIWF6bvvvlOLFi2Kff3+1NSf0alTJ+3atUvffvut0/i7774rm82mO++8s8TbdSVsNpt5lKXQ559/rv/973/XdL2F65ZUZP1z584t8TIL99eiRYucxt9///0/vayqVavqb3/7m5544gn99ttvRe7qu3gdS5cuVUFBgflzcy22rzR169ZNbm5u+umnn/7w5744N998s8aNG6fGjRsX+bmWpIoVKyoiIkKvvvqqJJk1NptNbm5uqlixolmbk5NT5Nll7dq1k3Th7rff+/DDD4vcORYVFaUdO3aoXr16xfb/+0BUyMfHRz169NCzzz6rvLw87dy583K7Ci7GESKUa1OnTlXz5s119OhR8/C1dOEfxYceekjz589XvXr11LRpU23atKlEf9Cu1NGjR9W3b18NHjxYDodDEyZMUKVKlcw7mKQLf8R69Oihbt26aeDAgbrhhhv022+/affu3fr222/1r3/9q0Trfuqpp/Tuu+/qrrvu0uTJk1WrVi19/vnneu211/T444/r5ptvvlqbWayoqCglJSWpQYMGatKkidLT0zV9+nTVrFnzmq5Xkho0aKB69erp6aeflmEY8vPz0/Lly//SaYquXbuqXbt2io+P1+nTp9WiRQutX7/+ih8I2qtXL4WHh6tFixYKCAjQgQMHNGvWLNWqVUthYWFOtR9//LHc3NzUpUsX8y6zpk2bmte2tGnTRtWqVdOwYcM0YcIE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4Prk5sqVL1myROnp6dqyZUuReZmZmZKkoKAgp/GgoCAdOHDArPHw8HA6slRYU/j+zMxMBQYGFll+YGCgWVOcxMRETZo06c9tEAAAKJdcdoTo4MGD+vvf/65FixapUqVKf1hns9mcpg3DKDJ2sYtriqu/3HLGjh0rh8Nhvg4ePHjJdQIAgPLLZYEoPT1dR48eVfPmzeXm5iY3NzetW7dOr7zyitzc3MwjQxcfxTl69Kg5Lzg4WHl5ecrKyrpkzZEjR4qs/9ixY0WOPv2ep6enfH19nV4AAOD65LJA1KlTJ23fvl0ZGRnmq0WLFnrwwQeVkZGhunXrKjg4WCkpKeZ78vLytG7dOrVp00aS1Lx5c7m7uzvVHD58WDt27DBrIiMj5XA4tGnTJrMmLS1NDofDrAEAANbmsmuIqlSpovDwcKcxHx8f+fv7m+NxcXFKSEhQWFiYwsLClJCQIG9vb0VHR0uS7Ha7Bg0apFGjRsnf319+fn4aPXq0GjdubF6k3bBhQ3Xv3l2DBw/W3LlzJUlDhgxRVFSU6tevX4pbDAAAyiqXXlR9OfHx8crJyVFsbKyysrIUERGhVatWqUqVKmbNSy+9JDc3N/Xr1085OTnq1KmTkpKSVLFiRbNm0aJFGjFihHk3Wu/evTVnzpxS3x4AAFA22QzDMFzdRHmQnZ0tu90uh8NxTa8nav6Pd6/ZsoHyKn36w65uAUA5daV/v13+HCIAAABXIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxABAADLIxAB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LI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALM+l33YPAFby8+TGrm4BKHNuHL/d1S1I4ggRAAAAgQgAAIBABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALI9ABAAALM9Sgei1115TnTp1VKlSJTVv3lz/+c9/XN0SAAAoAywTiD744APFxcXp2Wef1datW9W2bVv16NFDP//8s6tbAwAALmaZQDRz5kwNGjRIjz32mBo2bKhZs2YpNDRUr7/+uqtbAwAALmaJQJSXl6f09HR17drVabxr167asGGDi7oCAABlhZurGygNv/76q86dO6egoCCn8aCgIGVmZhb7ntzcXOXm5prTDodDkpSdnX3tGpV0Ljfnmi4fKI+u9e9daTl59pyrWwDKnGv9+124fMMwLllniUBUyGazOU0bhlFkrFBiYqImTZpUZDw0NPSa9Abgj9lnD3N1CwCulUR7qazm5MmTstv/eF2WCETVq1dXxYoVixwNOnr0aJGjRoXGjh2rkSNHmtPnz5/Xb7/9Jn9//z8MUbh+ZGdnKzQ0VAcPHpSvr6+r2wFwFfH7bS2GYejkyZMKCQm5ZJ0lApGHh4eaN2+ulJQU9e3b1xxPSUnR3XffXex7PD095enp6TRWtWrVa9kmyiBfX1/+wQSuU/x+W8eljgwVskQgkqSRI0cqJiZGLVq0UGRkpN588039/PPPGjaMQ/EAAFidZQLR/fffr+PHj2vy5Mk6fPiwwsPDtXLlStWqVcvVrQEAABezTCCSpNjYWMXGxrq6DZQDnp6emjBhQpHTpgDKP36/URybcbn70AAAAK5zlngwIwAAwKUQiAAAgOURiAAAgOURiHDd69Chg+Li4lzdBgCgDCMQAQAAyyMQAQAAyyMQwRLOnz+v+Ph4+fn5KTg4WBMnTjTnzZw5U40bN5aPj49CQ0MVGxurU6dOmfOTkpJUtWpVrVixQvXr15e3t7f+9re/6fTp01qwYIFq166tatWqafjw4Tp3jm8zB66lDz/8UI0bN5aXl5f8/f3VuXNnnT59WgMHDlSfPn00adIkBQYGytfXV0OHDlVeXp753uTkZN1xxx2qWrWq/P39FRUVpZ9++smcv3//ftlsNi1dulRt27aVl5eXWrZsqe+//16bN29WixYtVLlyZXXv3l3Hjh1zxebjGiIQwRIWLFggHx8fpaWladq0aZo8ebJSUlIkSRUqVNArr7yiHTt2aMGCBVq9erXi4+Od3n/mzBm98sorWrJkiZKTk7V27Vrdc889WrlypVauXKmFCxfqzTff1IcffuiKzQMs4fDhw+rfv78effRR7d692/w9LHyc3tdff63du3drzZo1Wrx4sZYtW6ZJkyaZ7z99+rRGjhypzZs36+uvv1aFChXUt29fnT9/3mk9EyZM0Lhx4/Ttt9/Kzc1N/fv3V3x8vF5++WX95z//0U8//aTx48eX6rajFBjAda59+/bGHXfc4TTWsmVLY8yYMcXWL1261PD39zen33nnHUOS8eOPP5pjQ4cONby9vY2TJ0+aY926dTOGDh16lbsHUCg9Pd2QZOzfv7/IvAEDBhh+fn7G6dOnzbHXX3/dqFy5snHu3Llil3f06FFDkrF9+3bDMAxj3759hiTjrbfeMmsWL15sSDK+/vprcywxMdGoX7/+1doslBEcIYIlNGnSxGm6Ro0aOnr0qCRpzZo16tKli2644QZVqVJFDz/8sI4fP67Tp0+b9d7e3qpXr545HRQUpNq1a6ty5cpOY4XLBHD1NW3aVJ06dVLjxo113333ad68ecrKynKa7+3tbU5HRkbq1KlTOnjwoCTpp59+UnR0tOrWrStfX1/VqVNHkvTzzz87ref3/14EBQVJkho3buw0xu/69YdABEtwd3d3mrbZbDp//rwOHDignj17Kjw8XB999JHS09P16quvSpLy8/Mv+f4/WiaAa6NixYpKSUnRF198oUaNGmn27NmqX7++9u3bd8n32Ww2SVKvXr10/PhxzZs3T2lpaUpLS5Mkp+uMJOff98L3XjzG7/r1x1Jf7gpcbMuWLSooKNCMGTNUocKF/z9YunSpi7sC8EdsNptuv/123X777Ro/frxq1aqlZcuWSZK+++475eTkyMvLS5KUmpqqypUrq2bNmjp+/Lh2796tuXPnqm3btpKkb775xmXbgbKHQARLq1evngoKCjR79mz16tVL69ev1xtvvOHqtgAUIy0tTV9//bW6du2qwMBApaWl6dixY2rYsKG2bdumvLw8DRo0SOPGjdOBAwc0YcIEPfnkk6pQoYKqVasmf39/vfnmm6pRo4Z+/vlnPf30067eJJQhnDKDpd16662aOXOmpk6dqvDwcC1atEiJiYmubgtAMXx9ffXvf/9bPXv21M0336xx48ZpxowZ6tGjhySpU6dOCgsLU7t27dSvXz/16tXLfMRGhQoVtGTJEqWnpys8PFxPPfWUpk+f7sKtQVljM4z/f78iAADl1MCBA3XixAl98sknrm4F5RRHiAAAgOURiAAAgOVxygwAAFgeR4gAAIDlEYgAAIDlEYgAAIDlEYgAAIDlEYgAWFZSUpKqVq36l5djs9l4/g1QzhGIAJRrAwcOVJ8+fVzdBoByjkAEAAAsj0AE4Lo1c+ZMNW7cWD4+PgoNDVVsbKxOnTpVpO6TTz7RzTffrEqVKqlLly46ePCg0/zly5erefPmqlSpkurWratJkyapoKCgtDYDQCkgEAG4blWoUEGvvPKKduzYoQULFmj16tWKj493qjlz5oyef/55LViwQOvXr1d2drYeeOABc/6XX36phx56SCNGjNCuXbs0d+5cJSUl6fnnny/tzQFwDfGkagDl2p/5Us9//etfevzxx/Xrr79KunBR9SOPPKLU1FRFRERIkvbs2aOGDRsqLS1NrVq1Urt27dSjRw+NHTvWXM57772n+Ph4HTp0SNKFi6qXLVvGtUxAOebm6gYA4FpZs2aNEhIStGvXLmVnZ6ugoEBnz57V6dOn5ePjI0lyc3NTixYtzPc0aNBAVatW1e7du9WqVSulp6dr8+bNTkeEzp07p7Nnz+rMmTPy9vYu9e0CcPURiABclw4cOKCePXtq2LBh+uc//yk/Pz998803GjRokPLz851qbTZbkfcXjp0/f16TJk3SPffcU6SmUqVK16Z5AKWOQATgurRlyxYVFBRoxowZqlDhwuWSS5cuLVJXUFCgLVu2qFWrVpKkvXv36sSJE2rQoIEkqVmzZtq7d69uuumm0mseQKkjEAEo9xwOhzIyMpzGAgICVFBQoNmzZ6tXr15av3693njjjSLvdXd31/Dhw/XKK6/I3d1dTz75pFq3bm0GpPHjxysqKkqhoaG67777VKFCBW3btk3bt2/XlClTSmPzAJQC7jIDUO6tXbtWt912m9Nr/vz5mjlzpqZOnarw8HAtWrRIiYmJRd7r7e2tMWPGKDo6WpGRkfLy8tKSJUvM+d26ddOKFSuUkpKili1bqnXr1po5c6Zq1apVmpsI4BrjLjMAAGB5HCECAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACWRyACAACW9/8AxVqBJVO5RSkAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(df.v1)\n",
    "plt.xlabel('Label')\n",
    "plt.title('Number of ham and spam messages')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f9b0040",
   "metadata": {},
   "source": [
    "# 1) Create input and output vectors.\n",
    "# 2) Process the labels."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "51337df4",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.v2\n",
    "Y = df.v1\n",
    "le = LabelEncoder()\n",
    "Y = le.fit_transform(Y)\n",
    "Y = Y.reshape(-1,1)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1de017c0",
   "metadata": {},
   "source": [
    "# Split into training and test data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "bba56d7f",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3d03f1a0",
   "metadata": {},
   "source": [
    "# Process the data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "977f8cb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "max_words = 1000\n",
    "max_len = 150\n",
    "tok = Tokenizer(num_words=max_words)\n",
    "tok.fit_on_texts(X_train)\n",
    "sequences = tok.texts_to_sequences(X_train)\n",
    "sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfa7090f",
   "metadata": {},
   "source": [
    "# Create Model and add Layers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a4276b89",
   "metadata": {},
   "outputs": [],
   "source": [
    "def RNN():\n",
    "    inputs = Input(name='inputs',shape=[max_len])\n",
    "    layer = Embedding(max_words,50,input_length=max_len)(inputs)\n",
    "    layer = LSTM(128)(layer)\n",
    "    layer = Dense(256,name='FC1')(layer)\n",
    "    layer = Activation('relu')(layer)\n",
    "    layer = Dropout(0.5)(layer)\n",
    "    layer = Dense(1,name='out_layer')(layer)\n",
    "    layer = Activation('tanh')(layer)\n",
    "    model = Model(inputs=inputs,outputs=layer)\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b3b3d850",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"model\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " inputs (InputLayer)         [(None, 150)]             0         \n",
      "                                                                 \n",
      " embedding (Embedding)       (None, 150, 50)           50000     \n",
      "                                                                 \n",
      " lstm (LSTM)                 (None, 128)               91648     \n",
      "                                                                 \n",
      " FC1 (Dense)                 (None, 256)               33024     \n",
      "                                                                 \n",
      " activation (Activation)     (None, 256)               0         \n",
      "                                                                 \n",
      " dropout (Dropout)           (None, 256)               0         \n",
      "                                                                 \n",
      " out_layer (Dense)           (None, 1)                 257       \n",
      "                                                                 \n",
      " activation_1 (Activation)   (None, 1)                 0         \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 174,929\n",
      "Trainable params: 174,929\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model = RNN()\n",
    "model.summary()\n",
    "model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy','mse','mae'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ecd528f5",
   "metadata": {},
   "source": [
    "# Fit the model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "0d388a08",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "28/28 [==============================] - 23s 667ms/step - loss: 0.3048 - accuracy: 0.8900 - mse: 0.0850 - mae: 0.1581 - val_loss: 0.1166 - val_accuracy: 0.9518 - val_mse: 0.0386 - val_mae: 0.0999\n",
      "Epoch 2/10\n",
      "28/28 [==============================] - 17s 598ms/step - loss: 0.0949 - accuracy: 0.9823 - mse: 0.0215 - mae: 0.0900 - val_loss: 0.0862 - val_accuracy: 0.9809 - val_mse: 0.0219 - val_mae: 0.0987\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x23a0538bd00>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.fit(sequences_matrix,Y_train,batch_size=128,epochs=10,\n",
    "          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "01f0df50",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_sequences = tok.texts_to_sequences(X_test)\n",
    "test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "93849fa3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "35/35 [==============================] - 2s 61ms/step - loss: 0.1076 - accuracy: 0.9803 - mse: 0.0235 - mae: 0.0974\n"
     ]
    }
   ],
   "source": [
    "accr = model.evaluate(test_sequences_matrix,Y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "a3ae19e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test set\n",
      "  Loss: 0.108\n",
      "  Accuracy: 0.980\n"
     ]
    }
   ],
   "source": [
    "print('Test set\\n  Loss: {:0.3f}\\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c186d04",
   "metadata": {},
   "source": [
    "# Save the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a051fcce",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.save(r\"C:\\Users\\ADMIN\\Downloads\\model_lSTM.h5\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2d575955",
   "metadata": {},
   "source": [
    "# Test the Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "43d44b15",
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import load_model\n",
    "m2 = load_model(r\"C:\\Users\\ADMIN\\Downloads\\model_lSTM.h5\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "a0e35774",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "35/35 [==============================] - 7s 165ms/step - loss: 0.0590 - accuracy: 0.9785 - mse: 0.0213 - mae: 0.0969\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "[0.058985039591789246,\n",
       " 0.9784753322601318,\n",
       " 0.021294036880135536,\n",
       " 0.09689562767744064]"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "m2.evaluate(test_sequences_matrix,Y_test)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
