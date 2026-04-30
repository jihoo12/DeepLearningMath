# covariance
두개의 변수가 어떻게 같이 변하는지
<br>
만약에 하나가 증가할때 다른 하나가 증가하면 positive covariance
<br>
만약에 하나가 감소할때 다른 하나가 증가하면 negative covariance

### positive covariance
두 변수가 같은 방향으로 움직이는것
### Negative covariance
두 변수가 반대로 움직이는것
### zero covariance
두 변수사이에 선형적인 관계(linear relationship) 이 없는것

### formula
$$cov(X, Y) = \frac{\sum (x_i - E[X])(y_i - E[Y])}{n}$$
- $x_i, y_i$ 는 각각 다른 점(data points)
- $E[X], E[Y]$ 는 값들의 평균
- n 점 개수