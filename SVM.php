<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<!--
	Future Imperfect by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
-->
<html>
	<head>
		<title>Final Project</title>
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no" />
		<link rel="stylesheet" href="assets/css/main.css" />
	</head>
	<body class="is-preload">
		
		<!-- Wrapper -->
			<div id="wrapper">

				<!-- Header -->
					<header id="header">
						<h1><a href = "#">SVM</a></h1>
					</header>
				
				<!-- Form -->
					<div id="main">
						<article class="post">
							<section>
								<form action="SVM.php" method="post">
										<div class="row gtr-uniform">
											<div class="col-12">
												<input type="text" name="review" id="review" value="" placeholder="Review" />
											</div>
											<div class="col-12">
												<ul class="actions">
													<li><input type="submit" value="Submit" /></li>
													<li><input type="reset" value="Reset" /></li>
												</ul>
											</div>
										</div>
									</form>
									<?php
										$review = $_POST["review"];
										if ($review != NULL) {
											$a='C:\Python38-32\python predict.py "'.$review.'"';
											exec($a,$out,$states);
											if ($out[0][1] == 1){
												echo $review." => Positive";
											}else{
												echo $review." => Negative";
											}
										}
									?>
							</section>
						</article>
					</div>

			</div>

		<!-- Scripts -->
			<script src="assets/js/jquery.min.js"></script>
			<script src="assets/js/browser.min.js"></script>
			<script src="assets/js/breakpoints.min.js"></script>
			<script src="assets/js/util.js"></script>
			<script src="assets/js/main.js"></script>
	</body>
</html>