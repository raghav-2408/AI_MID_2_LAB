# Mercury Tour

`action.moveToElement(we).build().perform();`

```java
package microProject;
public class practice {

	public static void main(String[] args)throws InterruptedException {
		System.setProperty("webdriver.chrome.driver", "path");
		WebDriver wb = new ChromeDriver();
		wb.manage().window().maximize();
		wb.get("https://www.mercurytravels.co.in/");
		Thread.sleep(1000);
		Actions action = new Actions(wb);
		WebElement we = wb.findElement(By.xpath("/html/body/nav[2]/div/div[2]/ul/li[1]/a"));
		action.moveToElement(we).build().perform();
		Thread.sleep(1000);
		WebElement we1 = wb.findElement(By.xpath("/html/body/nav[2]/div/div[2]/ul/li[1]/ul/li[2]/a"));
		we1.click();
		Thread.sleep(1000);
		WebElement we2 = wb.findElement(By.name("first_name"));
		we2.sendKeys("G");
		Thread.sleep(1000);
		WebElement ln = wb.findElement(By.name("last_name"));
		ln.sendKeys("Raghav");
		Thread.sleep(1000);
		WebElement we3 = wb.findElement(By.id("acc_user_email"));
		we3.sendKeys("ezioyt24@gmail.com");
		Thread.sleep(1000);
		WebElement we4 = wb.findElement(By.id("acc_user_password"));
		we4.sendKeys("Raghavraghav123");
		Thread.sleep(1000);
		WebElement we5 = wb.findElement(By.id("acc_user_passconf"));
		we5.sendKeys("Raghavraghav123");
		Thread.sleep(1000);
		WebElement we6 = wb.findElement(By.id("acc_mobile_no"));
		we6.sendKeys("9398545541");
		Thread.sleep(1000);
		WebElement we7 = wb.findElement(By.xpath("//*[@id=\"modalUserLogin\"]/div/div/div[2]/form/button"));
		we7.click();
		Thread.sleep(1000);
		
	}

}
```
# Facebook

`Select s1 = new Select(wb.findElement(By.name("birthday_month")));
		s1.selectByVisibleText("Aug");`
```java
package microProject;

public class practice {

	public static void main(String[] args)throws InterruptedException {
		System.setProperty("webdriver.chrome.driver", "path");
		WebDriver wb = new ChromeDriver();
		wb.manage().window().maximize();
		wb.get("https://www.facebook.com/");
		Thread.sleep(1000);
		WebElement we = wb.findElement(By.xpath("//a[text() = \"Create new account\"]"));
		we.click();
		Thread.sleep(1000);
//		firstname
		WebElement we1 = wb.findElement(By.name("firstname"));
		we1.sendKeys("G");
		Thread.sleep(1000);
		
		WebElement we2 = wb.findElement(By.name("lastname"));
		we2.sendKeys("Raghav");
		Thread.sleep(1000);
		
		WebElement we4 = wb.findElement(By.name("reg_email__"));
		we4.sendKeys("ezioyt24@gmail.com");
//		Thread.sleep(2000);
//		reg_email_confirmation__
		
		WebElement we6 = wb.findElement(By.name("reg_email_confirmation__"));
		we6.sendKeys("ezioyt24@gmail.com");
		Thread.sleep(1000);
		
		WebElement we5 = wb.findElement(By.name("reg_passwd__"));
		we5.sendKeys("Raghavraghav123");
		Thread.sleep(1000);
		
		Select s = new Select(wb.findElement(By.name("birthday_day")));
		s.selectByVisibleText("24");
		
		Select s1 = new Select(wb.findElement(By.name("birthday_month")));
		s1.selectByVisibleText("Aug");
		
		Select s2 = new Select(wb.findElement(By.name("birthday_year")));
		s2.selectByVisibleText("2002");
		
		WebElement wb7 = wb.findElement(By.xpath("//label[text() = \"Male\"]"));
		wb7.click();
		Thread.sleep(1000);
		
		WebElement wb8 = wb.findElement(By.name("websubmit"));
		wb8.click();
		
	}

}
```

# Myntra 

#### As inspect wont work, we use the following :
<br>

`Cntrl + Shift + I : Inspect`
<br>

`Cntrl + Shift + C : To Select an element`

```java
package microProject;
public class practice {

	public static void main(String[] args)throws InterruptedException {
		System.setProperty("webdriver.chrome.driver", "path");
		WebDriver wb = new ChromeDriver();
		wb.manage().window().maximize();
		
		wb.get("https://www.myntra.com/?utm_source=dms_google&utm_medium=searchbrand_cpc&utm_campaign=dms_google_searchbrand_cpc_Search_Brand_Myntra_Brand_India_BM_TROAS_SOK_New&gclid=Cj0KCQiA7aSsBhCiARIsALFvovwnyDJuvJi4pw9Dt1DtGMnPtM6YH8l_5alTydbwm3lZ70Nf788MneUaAi3UEALw_wcB");
		Thread.sleep(1000);
		
		WebElement we = wb.findElement(By.className("desktop-userTitle"));
		we.click();
		Thread.sleep(1000);
		
		WebElement we1 = wb.findElement(By.className("desktop-linkButton"));
		we1.click();
		Thread.sleep(1000);
		
		WebElement we2 = wb.findElement(By.className("mobileNumberInput"));
		we2.sendKeys("9398545541");
		Thread.sleep(1000);
		
		WebElement we3 = wb.findElement(By.className("submitBottomOption"));
		we3.click();	
	}
}
```

# AJIO

`Rememeber (select Facebook)`
```java
Set<String> parent = wb.getWindowHandles();
		Iterator iter = parent.iterator();
		while(iter.hasNext())
		{
			String child = (String)iter.next();
			if(!parent.equals(child))
			{
				wb.switchTo().window(child);
			}
		}
```

```java
package microProject;

public class practice {

	public static void main(String[] args)throws InterruptedException {
		System.setProperty("webdriver.chrome.driver", "path");
		WebDriver wb = new ChromeDriver();
		wb.manage().window().maximize();
		
		wb.get("https://www.ajio.com/?gad_source=1&gclid=Cj0KCQiA7aSsBhCiARIsALFvovy9k3WwxZupBml8YAfy-klJWx8atgnEn9g-6-ZawRIPN04vgdpE_HkaAnpBEALw_wcB");
		Thread.sleep(1000);
		
		WebElement we = wb.findElement(By.id("loginAjio"));
		we.click();
		Thread.sleep(1000);
		
		WebElement we1 = wb.findElement(By.className("social-txt"));
		we1.click();
		Thread.sleep(1000);
		
		Set<String> parent = wb.getWindowHandles();
		Iterator iter = parent.iterator();
		while(iter.hasNext())
		{
			String child = (String)iter.next();
			if(!parent.equals(child))
			{
				wb.switchTo().window(child);
			}
		}
		
		
		WebElement we2 = wb.findElement(By.name("email"));
		we2.sendKeys("ezioyt24@gmail.com");
		
		WebElement we3 = wb.findElement(By.name("pass"));
		we3.sendKeys("raghavraghav123");
		
		WebElement we4 = wb.findElement(By.name("login"));
		we4.click();
	}

}

```

# PDF from Word

### Choose ()[1] -> [1] should be outside of () 
`		WebElement we1 = wb.findElement(By.xpath("(//h3[@class = 'LC20lb MBeuO DKV0Md'])[1]"));
`

### Remember :
```java
Clipboard cp = Toolkit.getDefaultToolkit().getSystemClipboard();
		StringSelection ss = new StringSelection("Downloads\\SDE.doc");
		Thread.sleep(1000);
		cp.setContents(ss, null);
```

```java
package microProject;

public class practice {

	public static void main(String[] args)throws InterruptedException, AWTException {
		System.setProperty("webdriver.chrome.driver", "path");
		WebDriver wb = new ChromeDriver();
		wb.manage().window().maximize();
		wb.get("https://www.google.co.in/");
		WebElement we = wb.findElement(By.name("q"));
		we.sendKeys("word to pdf");
		we.submit();
		
		WebElement we1 = wb.findElement(By.xpath("(//h3[@class = 'LC20lb MBeuO DKV0Md'])[2]"));
		we1.click();
		
		WebElement we2 = wb.findElement(By.xpath("(//span[normalize-space() = 'Choose Files'])[1]"));
		we2.click();
		
		Clipboard cp = Toolkit.getDefaultToolkit().getSystemClipboard();
		StringSelection ss = new StringSelection("Downloads\\SDE.doc");
		Thread.sleep(1000);
		cp.setContents(ss, null);
		
		Robot robot = new Robot();
		robot.keyPress(KeyEvent.VK_CONTROL);
		robot.keyPress(KeyEvent.VK_V);
		robot.keyPress(KeyEvent.VK_ENTER);
		Thread.sleep(15000);
		
		WebElement we3 = wb.findElement(By.xpath("//span[text() = 'Download']"));
		we3.click();
}	
}
```

# GMAIL 

```java

```

```java
public class exp10 {
	public static void main(String[] args) throws InterruptedException {
System.setProperty("webdriver.chrome.driver", "path");
		WebDriver driver = new ChromeDriver();
		driver.get("http://gmail.com/");
		Thread.sleep(2000);
		WebElement createAccount = driver.findElement(By.xpath("(//span[normalize-space()='Create account'])[1]"));
		createAccount.click();
		Thread.sleep(2000);
		WebElement mySelft = driver.findElement(By.xpath("(//span[normalize-space()='For my personal use'])[1]"));
		mySelft.click();
		Thread.sleep(2000);
		WebElement firstName = driver.findElement(By.name("firstName"));
		firstName.sendKeys("Kumar01");
		Thread.sleep(2000);
		WebElement lastName = driver.findElement(By.name("lastName"));
		lastName.sendKeys("Ravi");
		Thread.sleep(2000);
		WebElement bn1 = driver.findElement(By.xpath("(//span[normalize-space()='Next'])"));
		bn1.click();
		Thread.sleep(3000);
		  Select month = new Select(driver.findElement(By.xpath("(//select[@id='month'])[1]")));
		  month.selectByValue("8");
		  Thread.sleep(2000);
		  WebElement day = driver.findElement(By.xpath("(//input[@id='day'])[1]"));
		  day.sendKeys("24");
		  Thread.sleep(2000);
		  WebElement year = driver.findElement(By.xpath("(//input[@id='year'])[1]"));
		  year.sendKeys("1996");
		  Select gender = new
		  Select(driver.findElement(By.xpath("(//select[@id='gender'])[1]")));
		  gender.selectByValue("1");
		  Thread.sleep(2000);
		  WebElement bn2 = driver.findElement(By.xpath("(//span[normalize-space()='Next'])"));
		  bn2.click();
		  Thread.sleep(2000);
		  WebElement uid = driver.findElement(By.name("Username"));
		  uid.sendKeys("Kmrthakla");
		  WebElement bn3 = driver.findElement(By.xpath("(//span[normalize-space()='Next'])"));
		  bn3.click();
		  Thread.sleep(2000);
		  WebElement pswd = driver.findElement(By.name("Passwd"));
		  pswd.sendKeys("Cmrit123#");
		  WebElement cpswd = driver.findElement(By.name("PasswdAgain"));
		  cpswd.sendKeys("Cmrit123#");
		  WebElement bn4 = driver.findElement(By.xpath("(//span[normalize-space()='Next'])"));
		  bn4.click();	
	}
}
```
